import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed

from split_dataset import Blocks, SplitDataset, EmptySplitDataset
from fimpy.roi_extraction import corr_algorithms as ca


def _corr_map_plane(dataset, block, out_file, time_lims, window_size):
    if time_lims is None:
        time_slice = slice(None)
    else:
        time_slice = slice(*time_lims)
    vid = dataset[(time_slice,) + Blocks.block_to_slices(block)]
    cmap = ca.correlation_map(vid, window_size)
    fl.save(out_file, dict(stack_3D=cmap))


def correlation_map(
    dataset, output_dir=None, time_lims=None, window_size=(1, 3, 3), n_jobs=10
):
    new_dataset = EmptySplitDataset(
        shape_full=dataset.shape[1:],
        shape_block=(1,) + dataset.shape[2:],
        root=output_dir or dataset.root.parent,
        name="correlation_map",
    )
    Parallel(n_jobs=n_jobs)(
        delayed(_corr_map_plane)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            time_lims,
            window_size,
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )
    return new_dataset.finalize()
