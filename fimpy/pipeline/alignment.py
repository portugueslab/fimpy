import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed

from split_dataset import EmptySplitDataset, Blocks
from fimpy.alignment.volume import (
    find_shifts_sobel,
    shift_stack,
    sobel_stack,
)


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
