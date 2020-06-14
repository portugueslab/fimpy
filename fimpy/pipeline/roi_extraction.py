import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed

from split_dataset import Blocks, SplitDataset, EmptySplitDataset
from fimpy.roi_extraction import corr_algorithms as ca
from fimpy.roi_extraction.anatomical import extract_traces_around_points



def _corr_map_plane(dataset, block, out_file, time_lims, window_size):
    if time_lims is None:
        time_slice = slice(None)
    else:
        time_slice = slice(*time_lims)
    vid = dataset[(time_slice,) + Blocks.block_to_slices(block)]
    cmap = ca.correlation_map(vid, window_size)
    fl.save(out_file, dict(stack_3D=cmap))


def correlation_map(dataset, output_dir=None, time_lims=None, window_size=(1, 3, 3), n_jobs=10):
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


def _extract_rois_block(dataset, block, out_file, rois):
    vid = dataset[Blocks.block_to_slices(block)]
    traces = ca.extract_traces(vid, rois)
    fl.save(out_file, traces)


def _extract_traces_coords(dataset, block, out_file, coords, **kwargs):
    vid = dataset[Blocks.block_to_slices(block)]
    traces = extract_traces_around_points(vid, coords, **kwargs)
    fl.save(out_file, traces)


def extract_traces_coords(
    dataset: SplitDataset, coords, output_dir=None, block_duration=60, n_jobs=5, **kwargs
):
    new_dataset = EmptySplitDataset(
        shape_full=dataset.shape,
        shape_block=(block_duration,) + dataset.shape[1:],
        root=output_dir or dataset.root.parent,
        name="traces",
    )
    Parallel(n_jobs=n_jobs)(
        delayed(_extract_traces_coords)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            coords=coords,
            **kwargs
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )

    trace_dset = new_dataset.finalize()
    traces = np.concatenate(
        [fl.load(str(f), "/traces") for f in trace_dset.files.flatten()], 1
    )
    trace_data = dict(traces=traces, coords=coords)
    fl.save(str(trace_dset.root.parent / "traces.h5"), trace_data)

    return trace_data
