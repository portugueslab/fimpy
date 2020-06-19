import flammkuchen as fl
import numpy as np
from joblib import Parallel, delayed

from split_dataset import Blocks, SplitDataset, EmptySplitDataset
from fimpy.roi_extraction import corr_algorithms as ca
from fimpy.roi_extraction.anatomical import extract_traces_around_points
from fimpy.roi_extraction import merging


def _grow_block_rois(dataset, cmap_dataset, block, out_file, time_lims, **kwargs):
    vid = dataset[(slice(*time_lims),) + Blocks.block_to_slices(block)]
    cmap = cmap_dataset[Blocks.block_to_slices(block)]
    rois = ca.grow_rois(vid, cmap, **kwargs)
    fl.save(out_file, dict(stack_3D=rois))


def grow_rois(
    dataset,
    cmap,
    output_dir=None,
    blocksize=(2, 50, 50),
    padding=(2, 20, 20),
    time_lims=None,
    **kwargs
):
    new_dataset = EmptySplitDataset(
        shape_full=dataset.shape[1:],
        shape_block=blocksize,
        padding=padding,
        root=output_dir or dataset.root.parent,
        name="rois",
    )
    Parallel(n_jobs=10)(
        delayed(_grow_block_rois)(
            dataset,
            cmap,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            time_lims,
            **kwargs
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )
    rois = new_dataset.finalize()
    return _merge_rois(rois)


def _merge_rois(rois: SplitDataset):
    def load_array(x):
        return fl.load(str(x), "/stack_3D")

    vfunc = np.vectorize(load_array, otypes=[np.object])
    loaded = vfunc(rois.files)
    merged_rois = merging.merge_rois(rois, loaded)
    fl.save(str(rois.root.parent / "merged_rois.h5"), dict(stack_3D=merged_rois))
    return merged_rois


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


def _extract_rois_block(dataset, block, out_file, rois):
    vid = dataset[Blocks.block_to_slices(block)]
    traces = ca.extract_traces(vid, rois)
    fl.save(out_file, traces)


def _extract_traces_coords(dataset, block, out_file, coords, **kwargs):
    vid = dataset[Blocks.block_to_slices(block)]
    traces = extract_traces_around_points(vid, coords, **kwargs)
    fl.save(out_file, traces)


def extract_traces_coords(
    dataset: SplitDataset,
    coords,
    output_dir=None,
    block_duration=60,
    n_jobs=5,
    **kwargs
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


def extract_traces(dataset: SplitDataset, rois, output_dir=None, block_duration=40):
    new_dataset = EmptySplitDataset(
        shape_full=dataset.shape,
        shape_block=(block_duration,) + dataset.shape[1:],
        root=output_dir or dataset.root.parent,
        name="traces",
    )
    Parallel(n_jobs=20)(
        delayed(_extract_rois_block)(
            dataset,
            new_block,
            str(new_dataset.root / new_dataset.files[i_block]),
            rois=rois,
        )
        for i_block, (_, new_block) in enumerate(new_dataset.slices(as_tuples=True))
    )

    trace_dset = new_dataset.finalize()
    traces = np.concatenate(
        [fl.load(str(f), "/traces") for f in trace_dset.files.flatten()], 1
    )
    first_file = trace_dset.files.flatten()[0]
    coords = fl.load(str(first_file), "/coords")
    areas = fl.load(str(first_file), "/areas")
    trace_data = dict(traces=traces, coords=coords, areas=areas)
    fl.save(str(trace_dset.root.parent / "traces.h5"), trace_data)

    return trace_data
