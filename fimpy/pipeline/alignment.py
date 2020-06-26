import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed

from split_dataset import EmptySplitDataset, Blocks
from fimpy.alignment.volume import (
    find_shifts_sobel,
    shift_stack,
    sobel_stack,
)
from skimage.feature import register_translation
from fimpy.alignment.plane import align_single_planes_sobel


def align_volumes_with_filtering(
    dataset,
    output_dir=None,
    ref_window_halfsize=25,
    fft_reference=None,
    register_every=100,
    reg_halfwin=30,
    prefilter_sigma=3.3,
    block_size=120,
    n_jobs=10,
    verbose=False,
):
    """ Aligns a dataset with prefiltering, by taking averages

    :param dataset: the input dataset
    :param output_dir: (optional, not recommended) the output folder
    :param ref_window_halfsize: the length of the time-average taken for the reference
    :param fft_reference: (optional) a fourier transform of a reference stack,
        if not supplied, one will be calculated from the middle of the dataset
    :param register_every: how many frames apart are the points which will
        be registered to the reference
    :param reg_halfwin: the length of the time window to take the average for
        registration
    :param prefilter_sigma: the width of the filter for sobel-prefiltering before
        the alignment
    :param block_size: the duration (in frames) of the aligned blocks
    :return:
    """
    time_middle = dataset.shape[0] // 2

    # prepare the destination
    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="aligned",
        shape_full=dataset.shape,
        shape_block=(block_size,) + dataset.shape_block[1:],
    )

    # calculate the reference around the temporal middle of the dataset if
    # a reference is not provided.

    if fft_reference is None:
        fft_reference = np.fft.fftn(
            sobel_stack(
                np.mean(
                    dataset[
                        time_middle
                        - ref_window_halfsize : time_middle
                        + ref_window_halfsize,
                        :,
                        :,
                        :,
                    ],
                    0,
                ),
                prefilter_sigma,
            )
        )

    # set the frames at which the registration happens. Other shifts will
    # be interpolated
    if verbose:
        print("finding shifts...")
    shift_centres = range(reg_halfwin, dataset.shape[0] - reg_halfwin, register_every)
    shift_times = np.array(list(shift_centres))

    # find the shifts in parallel
    shifts = Parallel(n_jobs=n_jobs)(
        delayed(_get_shifts)(
            dataset, t - reg_halfwin, t + reg_halfwin, fft_reference, prefilter_sigma
        )
        for t in shift_centres
    )
    shifts = np.stack(shifts, 0)

    if verbose:
        print("Saving shifts...")
    # save the shifts
    fl.save(
        str(new_dataset.root / "shifts_sobel.h5"),
        dict(
            shift_times=shift_times,
            shifts=shifts,
            parameters=dict(
                ref_window_halfsize=ref_window_halfsize,
                fft_reference=fft_reference,
                register_every=register_every,
                reg_halfwin=reg_halfwin,
                prefilter_sigma=prefilter_sigma,
                block_size=block_size,
            ),
        ),
    )

    if verbose:
        print("Applying shifts...")
    # apply them in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(_apply_shifts)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            shifts,
            shift_times,
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )
    return new_dataset.finalize()


def _get_shifts(dataset, tstart, tend, fft_ref, prefilter_sigma):
    print("Finding shifts at ", tstart)
    return find_shifts_sobel(dataset[tstart:tend, :, :, :], fft_ref, prefilter_sigma)


def _apply_shifts(dataset, block, out_file, shifts, shift_times):
    vid = dataset[Blocks.block_to_slices(block)]
    aligned = shift_stack(vid, range(block[0][0], block[0][1]), shifts, shift_times)
    print(out_file)
    fl.save(out_file, dict(stack_4D=aligned, shifts=shifts))


def apply_shifts(dataset, output_dir=None, block_size=120, n_jobs=10, verbose=False):
    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="aligned",
        shape_full=dataset.shape,
        shape_block=(block_size,) + dataset.shape_block[1:],
    )

    shifts_data = fl.load(str(next(output_dir.glob("*shifts*"))))

    Parallel(n_jobs=n_jobs)(
        delayed(_apply_shifts)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            shifts_data["shifts"],
            shifts_data["shifts"],
            shifts_data["shift_times"],
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )
    return new_dataset.finalize()


def align_2p_volume(
    dataset,
    output_dir=None,
    reference=None,
    n_frames_ref=10,
    across_planes=None,
    prefilter_sigma=3.3,
    upsample_factor=10,
    max_shift=15,
    n_jobs=20,
    verbose=True,
):
    """ Function for complete alignment of two-photon, planar acquired stack

    :param dataset: input H5Dataset
    :param output_dir: optional, output destination directory, subdirectory aligned will appear
    :param reference: optional, reference to align to
    :param n_frames_ref: number of frames to take as reference mean, if reference is being calculated
    :param across_planes: bool, True by default if reference is not provided, whether to align across planes
    :param prefilter_sigma: feature size to filter for better alignment. if < 0 no filtering will take place
    :param upsample_factor: granularity of subpixel shift
    :param max_shift: maximum shift allowed
    :param n_jobs: number of parallel jobs
    :return: reference to align dataset
    """

    # prepare the destination
    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="aligned",
        shape_full=dataset.shape,
        shape_block=(dataset.shape_block[0], 1) + dataset.shape_block[2:],
    )

    if verbose:
        print("Calculating filtered reference")
    if reference is None:
        t_mid = dataset.shape[0] // 2
        reference = dataset[t_mid : t_mid + n_frames_ref, :, :, :].mean(0)
        if across_planes is None:
            across_planes = False
    else:
        if across_planes is None:
            across_planes = True

    sob_ref = sobel_stack(reference, prefilter_sigma)

    n_planes = reference.shape[0]
    shifts_planes = np.zeros((n_planes, 2))

    centre_plane = int(n_planes // 2)

    if across_planes:
        if verbose:
            print("Registering across planes...")
        # Find between-planes shifts
        for i in range(centre_plane, reference.shape[0] - 1):
            s, _, _ = register_translation(
                reference[i, :, :], reference[i + 1, :, :], 10
            )
            shifts_planes[i + 1, :] = shifts_planes[i, :] + s
        for i in range(centre_plane, 0, -1):
            s, _, _ = register_translation(
                reference[i, :, :], reference[i - 1, :, :], 10
            )
            shifts_planes[i - 1, :] = shifts_planes[i, :] + s

    fl.save(dataset.root.parent / "shifts.h5", shifts_planes)

    if verbose:
        print("Aligning individual planes...")
    Parallel(n_jobs=n_jobs)(
        delayed(_align_and_shift)(
            dataset,
            new_block,
            sob_ref[i_block : i_block + 1, :, :],
            str(new_dataset.root / new_dataset.files[i_block]),
            shifts_planes[i_block, :],
            prefilter_sigma,
            upsample_factor,
            max_shift,
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )

    return new_dataset.finalize()


def _align_and_shift(
    dataset,
    block,
    ref,
    out_file,
    shift_plane,
    prefilter_sigma,
    upsample_factor,
    max_shift,
):
    stack = dataset[Blocks.block_to_slices(block)]
    shifted, shifts = align_single_planes_sobel(
        stack,
        np.fft.fftn(ref),
        prefilter_sigma=prefilter_sigma,
        upsample_factor=upsample_factor,
        maxshift=max_shift,
        offset=-shift_plane,
    )
    fl.save(out_file, dict(stack_4D=shifted, shifts=shifts), compression="blosc")

    print("Saved {}...".format(out_file))

