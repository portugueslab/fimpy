from functools import wraps
from pathlib import Path
import numpy as np

import flammkuchen as fl
from joblib import Parallel, delayed

from split_dataset import SplitDataset, Blocks, EmptySplitDataset


def h5cache(func, output_dir: Path, filename=None):
    @wraps(func)
    def return_cached(*args, **kwargs):
        f_name = filename or func.__name__
        output_file = output_dir / (f_name + ".h5")
        if output_file.is_file():
            return fl.load(output_file)
        else:
            res = func(*args, **kwargs)
            fl.save(output_file, res)
            return res

    return return_cached


def run_in_blocks(
    function,
    dataset: SplitDataset,
    *extra_args,
    per_block_args=None,
    output_dir=None,
    output_shape_full=None,
    output_shape_block=None,
    process_shape_block=None,
    n_jobs=20,
    output_name=None,
    **kwargs
):
    """

    Runs a function over a split dataset in parallel

    :param function: the function to be applied (e.g. delta f over f or regression)
    :param dataset: the split dataset
    :param extra_args: the other positional arguments to the function
    :param per_block_args: a dictionary or list of extra arguments
    :param output_dir: (optional) the output directory
    :param output_shape_full: the output shape, if it will be different\
    :param process_shape_block: the size of block to process
    :param output_shape_block: the output block size, if different
    :param n_jobs: number of jobs to parallelize to
    :param output_name: the name of the output dataset, the function name is used
        if left blank
    :param kwargs: extra keyword arguments to the function
    :return: the processed dataset
    """

    # TODO avoid duplication of execution on first block
    # TODO figure out output_shape_full
    process_shape_block = process_shape_block or dataset.shape_block

    # Automatically determine the output shape
    processing_blocks = Blocks(
        shape_full=dataset.shape_full, shape_block=process_shape_block
    )

    _, new_block = list(processing_blocks.slices(as_tuples=True))[0]

    if output_shape_block is None:
        processed = function(
            dataset[Blocks.block_to_slices(new_block)],
            *extra_args,
            *([] if per_block_args is None else per_block_args[0]),
            **kwargs
        )
        output_shape_block = processed.shape

    new_dataset = EmptySplitDataset(
        root=output_dir or dataset.root.parent,
        name=output_name or function.__name__,
        shape_full=output_shape_full or dataset.shape,
        shape_block=output_shape_block or process_shape_block,
        resolution=dataset.resolution,
    )

    def wrap_function(ds, *args, filename, new_block, **kwargs):
        original = ds[Blocks.block_to_slices(new_block)]
        processed = function(original, *args, **kwargs)
        fl.save(filename, {"stack_{}D".format(processed.ndim): processed})

    Parallel(n_jobs=n_jobs)(
        delayed(wrap_function)(
            dataset,
            *extra_args,
            *([] if per_block_args is None else per_block_args[i_block]),
            new_block=new_block,
            filename=str(new_dataset.root / new_dataset.files[i_block]),
            **kwargs
        )
        for i_block, ((_, new_block)) in enumerate(
            processing_blocks.slices(as_tuples=True)
        )
    )
    return new_dataset.finalize()
