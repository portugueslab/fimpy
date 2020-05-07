import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed
from skimage.measure import block_reduce

from split_dataset import Blocks, SplitDataset, EmptySplitDataset


def _time_percentile(
    dataset, block, out_file, method="mean", percentile=50, time_slice=None
):
    if time_slice is None:
        time_slice = slice(None)
    else:
        time_slice = slice(*time_slice)

    vid = dataset[(time_slice,) + Blocks.block_to_slices(block)]
    if method == "percentile":
        fl.save(out_file, dict(stack_3D=np.percentile(vid, percentile, 0)))
    elif method == "mean":
        fl.save(out_file, dict(stack_3D=np.mean(vid, 0)))
    else:
        raise AssertionError(f"Invalid method {method}")


def make_anatomy(dataset: SplitDataset, output_dir=None, block_size=None, **kwargs):
    """ Make an anatomy stack from a 4D dataset

    :param dataset:
    :param output_dir:
    :param block_size:
    :param kwargs:
        method: str
            either "mean" or "percentile"
        percentile:
            if method is "percentile", which percentile to take for anatomy
        time_slice: Optional[tuple[int, int]]
            if set, take only the time in the given interval for the anatomy

    :return:
    """
    new_dataset = EmptySplitDataset(
        shape_full=dataset.shape[1:],
        shape_block=(block_size if block_size else dataset.shape_block[1:]),
        root=output_dir or dataset.root.parent,
        name="anatomy",
    )
    Parallel(n_jobs=20)(
        delayed(_time_percentile)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            **kwargs,
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )
    return new_dataset.finalize()


def calc_f0(stack, frames):
    """ Calculate the baseline flourescence over
    a chosen list of frames

    :param stack:
    :param frames: an iterable of baseline frames
    :return:
    """
    fr_mean = None
    for i_frame in frames:
        sf = stack[int(i_frame), :, :, :]
        if fr_mean is None:
            fr_mean = sf
        else:
            fr_mean += sf
    return fr_mean / len(frames)


def _dff(
    dataset,
    block,
    dest_filename,
    baseline_stack,
    multiplier=128,
    output_type=np.int16,
    subtract=0,
):
    stack = dataset[Blocks.block_to_slices(block)]
    baseline_sel = baseline_stack[
        Blocks.block_to_slices(block)[1:]
    ]  # crop the corresponding slice of the baseline
    dffi = (
        multiplier * (stack - baseline_sel) / np.maximum(baseline_sel - subtract, 1)
    ).astype(output_type)
    fl.save(dest_filename, dict(stack_4D=dffi), compression="blosc")
    return None


def dff(dataset: SplitDataset, baseline_stack, output_dir=None, n_jobs=20, **kwargs):
    """ Calculates change over baseline
    :param dataset:
    :param baseline_stack: F stack for the (F_i - F) / F calculation
    :param output_dir:
    :param n_jobs:
    :return:
    """
    old_dataset = Blocks(shape_full=dataset.shape, shape_block=dataset.shape_block)

    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="dff",
        shape_full=dataset.shape,
        shape_block=dataset.shape_block,
    )

    Parallel(n_jobs=n_jobs)(
        delayed(_dff)(
            dataset,
            old_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            baseline_stack,
            **kwargs,
        )
        for i_block, (_, old_block) in enumerate(old_dataset.slices(as_tuples=True))
    )

    return new_dataset.finalize()


def _downsample_block(dataset, old_block, filename, factor, method):
    original = dataset[Blocks.block_to_slices(old_block)]
    downsampled = block_reduce(original, factor, method)
    ndims = len(downsampled.shape)
    fl.save(filename, {f"stack_{ndims}D": downsampled})


def downsample(
    dataset: SplitDataset,
    downsampling=2,
    proc_block_shape=None,
    crop=None,
    output_dir=None,
    n_jobs=20,
    method=np.sum,
):
    """ Donwsamples a dataset

    :param dataset:
    :param downsampling:
    :param crop:
    :param output_dir:
    :param n_jobs:
    :param method:
    :return:
    """
    crop = crop or tuple((0, 0) for _ in dataset.shape)
    old_dataset = Blocks(
        shape_full=dataset.shape,
        crop=crop,
        shape_block=proc_block_shape if proc_block_shape else dataset.shape_block,
    )

    shape_downsampled = tuple(
        sc // ds for sc, ds in zip(old_dataset.shape_cropped, downsampling)
    )
    block_size_downsampled = tuple(
        sb // ds for (sb, ds) in zip(old_dataset.shape_block, downsampling)
    )

    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="downsampled",
        shape_full=shape_downsampled,
        shape_block=block_size_downsampled,
    )

    Parallel(n_jobs=n_jobs)(
        delayed(_downsample_block)(
            dataset,
            old_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            downsampling,
            method,
        )
        for i_block, (_, old_block) in enumerate(old_dataset.slices(as_tuples=True))
    )

    return new_dataset.finalize()


def _average_block(dataset, block, start, trial_duration, n_trials, out_file):
    vid = dataset[
        (slice(start, start + trial_duration * n_trials),)
        + Blocks.block_to_slices(block)[1:]
    ]
    vid_trials = np.reshape(vid, (n_trials, trial_duration) + vid.shape[1:])

    fl.save(out_file, dict(stack_4D=np.sum(vid_trials.astype(np.uint32), 0)))


def average_trials(
    dataset, start, trial_duration, n_trials, block_size, output_dir=None
):
    """ Averages dataset over trials

    :param dataset: input dataset
    :param destination: output folder
    :param start:
    :param trial_duration:
    :param n_trials:
    :param block_size:
    :return:
    """
    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name="trial_averages",
        shape_full=(trial_duration,) + dataset.shape[1:],
        shape_block=(trial_duration,) + block_size,
    )
    Parallel(n_jobs=20)(
        delayed(_average_block)(
            dataset,
            new_block,
            start,
            trial_duration,
            n_trials,
            str(new_dataset.root / new_dataset.files[i_block]),
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )

    new_dataset.finalize()
