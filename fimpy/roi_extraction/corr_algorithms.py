from numba import jit
import numpy as np
import math
from fimpy.utilities import fast_pearson


@jit(nopython=True)
def _correlation_map_jit(video, wrs, neighbors_a, neighbors_b):
    """ Function that computes the actual voxelwise correlation map. Depends
    on some funny inputs that need to be kept out of numba function.
    Exclude from the calculation invalid timepoints (with 0 value).

    :param video: 4D stack
    :param wrs: tuple with number of voxels to include along each dimension;
    :param neighbors_a: first half of voxels neighbors
    :param neighbors_b: second half of voxels neighbors
    :return: correlation map (3D float)
    """
    cm = np.zeros(video.shape[1:])

    n_neighbors = len(neighbors_a)

    cors = np.zeros(video.shape[1:] + (n_neighbors,))

    # First, store the correlation for half of the neighbours in each voxel
    # in a temporary array.
    # Loop over coordinates:
    for z in range(wrs[0], video.shape[1] - wrs[0]):
        for y in range(wrs[1], video.shape[2] - wrs[1]):
            for x in range(wrs[2], video.shape[3] - wrs[2]):
                # Loop over first half of neighbours:
                for i_n, delta in enumerate(neighbors_a):
                    # To avoid bad points that are set to 0,
                    # exclude them from the fast_pearson computing:
                    focal_trace = video[:, z, y, x]
                    neighbor_trace = video[:, z + delta[0], y + delta[1], x + delta[2]]
                    valid_pixels = (focal_trace > 0) & (neighbor_trace > 0)

                    # Calculate pearson if there are at least 3 valid timepts:
                    if np.sum(valid_pixels) > 3:
                        cors[z, y, x, i_n] = fast_pearson(
                            focal_trace[valid_pixels], neighbor_trace[valid_pixels]
                        )
                    else:
                        cors[z, y, x, i_n] = 0

    # Now use the stored correlations to compute the final correlations,
    # using twice each correlation for pair of pixels:
    for z in range(wrs[0], video.shape[1] - wrs[0]):
        for y in range(wrs[1], video.shape[2] - wrs[1]):
            for x in range(wrs[2], video.shape[3] - wrs[2]):
                # sum the correlations with half of its neighbors
                cm[z, y, x] += np.sum(cors[z, y, x, :])
                # and for the other half, pick the correlations
                # corresponding to correlations with this coordinate
                for i_n, delta in enumerate(neighbors_b):
                    cm[z, y, x] += cors[
                        z + delta[0], y + delta[1], x + delta[2], n_neighbors - i_n
                    ]
    # We have summed correlations, now divide to get mean:
    cm /= n_neighbors * 2
    return cm


@jit(nopython=True)
def _tuple3_add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2])


@jit(nopython=True)
def _tuple3_square(t):
    return (t[0] ** 2, t[1] ** 2, t[2] ** 2)


@jit(nopython=True)
def _tuple3_dist2(t1, t2):
    """ Squared distance between tuples

    :param t1:
    :param t2:
    :return:
    """
    return (t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2 + (t1[2] - t2[2]) ** 2


@jit(nopython=True)
def _dist_vox(t1, t2, vs2):
    """ Squared distance betwwen tuples,
    taking into account voxes sizes

    :param t1:
    :param t2:
    :return:
    """
    return (
        vs2[0] * ((t1[0] - t2[0]) ** 2)
        + vs2[1] * ((t1[1] - t2[1]) ** 2)
        + vs2[2] * ((t1[2] - t2[2]) ** 2)
    )


@jit(nopython=True, cache=True)
def _find_neighbours(indexes, maxdim, only_2D=False):
    neighbours = set()
    if indexes[0] > 0 and not only_2D:
        neighbours.add(_tuple3_add(indexes, (-1, 0, 0)))
    if indexes[1] > 0:
        neighbours.add(_tuple3_add(indexes, (0, -1, 0)))
    if indexes[2] > 0:
        neighbours.add(_tuple3_add(indexes, (0, 0, -1)))
    if indexes[0] < maxdim[0] - 1 and not only_2D:
        neighbours.add(_tuple3_add(indexes, (1, 0, 0)))
    if indexes[1] < maxdim[1] - 1:
        neighbours.add(_tuple3_add(indexes, (0, 1, 0)))
    if indexes[2] < maxdim[2] - 1:
        neighbours.add(_tuple3_add(indexes, (0, 0, 1)))
    return neighbours


@jit(nopython=True)
def _n_unravel_2(i, shape):
    return i // shape[1], i % shape[1]


@jit(nopython=True)
def _n_unravel_3(i, shape):
    z = i // (shape[1] * shape[2])
    xy = i % (shape[1] * shape[2])
    y = xy // shape[2]
    x = xy % shape[2]
    return (z, y, x)


@jit(nopython=True)
def _is_eligible(
    idx,
    root_idx,
    trace_so_far,
    video,
    init_corr_thr=0.15,
    final_corr_thr=0.15,
    max_r2=100,
    corr_thresh_inc_dist=1,
    voxel_size2=(1, 1, 1),
):
    distance2 = _dist_vox(idx, root_idx, voxel_size2)
    if distance2 > max_r2:
        return False
    threshold = (
        init_corr_thr
        + (final_corr_thr - init_corr_thr)
        * min(math.sqrt(distance2), corr_thresh_inc_dist)
        / corr_thresh_inc_dist
    )
    trace_to_test = video[(slice(0, video.shape[0]),) + idx] * 1.0
    corr = fast_pearson(trace_to_test, trace_so_far)
    if corr < threshold:
        return False
    return True


@jit(nopython=True, cache=True)
def _jit_flood(
    video,
    corr_map,
    init_corr_thr=0.15,
    max_labels=100,
    final_corr_thr=0,
    max_radius=10,
    corr_thresh_inc_dist=1,
    max_investigate=2000,
    min_area=1,
    across_planes=True,
    voxel_size=(1, 1, 1),
):
    if final_corr_thr < init_corr_thr:
        final_corr_thr = init_corr_thr

    if across_planes:
        n_parts = 1
    else:
        n_parts = video.shape[1]

    i_roi = 0

    max_r2 = max_radius ** 2
    voxel_size2 = _tuple3_square(voxel_size)

    valid = corr_map.copy()
    labels = -np.ones(video.shape[1:], dtype=np.int32)
    trace_so_far = np.zeros(video.shape[0], dtype=np.float64)

    # this loop has one iteration if we build ROIs across planes
    for i_part in range(n_parts):
        i_roi_part = 0
        n_investigated = 0  # this counts ROIs detected in one plane
        while (i_roi_part < max_labels) and (n_investigated < max_investigate):
            # find the location of the labeled pixel
            n_investigated += 1

            selected_indexes = []
            if across_planes:
                root_idx = _n_unravel_3(np.argmax(valid), corr_map.shape)
            else:
                root_idx = (i_part,) + _n_unravel_2(
                    np.argmax(valid[i_part, :, :]), corr_map.shape[1:]
                )

            indexes_to_process = [root_idx]

            # flood fill, adding indexes to the selected indexes
            # if they satisfy the criteria

            while len(indexes_to_process) > 0:
                current_index = indexes_to_process.pop()

                n_sel = len(selected_indexes)
                if n_sel == 0 or (
                    valid[current_index] > 0
                    and _is_eligible(
                        current_index,
                        root_idx,
                        trace_so_far,
                        video,
                        init_corr_thr,
                        final_corr_thr,
                        max_r2,
                        corr_thresh_inc_dist,
                        voxel_size2,
                    )
                ):

                    # update the average trace for correlation
                    if n_sel == 0:
                        trace_so_far[:] = (
                            video[(slice(0, video.shape[0]),) + current_index] * 1.0
                        )
                    else:
                        trace_so_far[:] = (
                            trace_so_far[:] * n_sel
                            + video[(slice(0, video.shape[0]),) + current_index] * 1.0
                        ) / (n_sel + 1)
                    selected_indexes.append(current_index)
                    valid[current_index] = 0
                    indexes_to_process.extend(
                        _find_neighbours(
                            current_index, video.shape[1:], not across_planes
                        )
                    )

            # For the assembled ROI, check if the area is big engough
            if len(selected_indexes) >= min_area:
                # if so, fill out the ROI labeling array with a ROI id,
                # and make it inacessible to further ROI finding
                for idx in selected_indexes:
                    labels[idx] = i_roi
                    valid[idx] = 0
                i_roi += 1
                i_roi_part += 1
            else:
                # if the area is too small, mark these indexes as not
                # belonging to a ROI
                for idx in selected_indexes:
                    labels[idx] = -2
                    valid[idx] = 0

    return labels


@jit(nopython=True)
def _flood_fill(full_labels, new_labels, start, offset, val):
    idxs = [start]
    fill_val = new_labels[start]
    while len(idxs) > 0:
        ci = idxs.pop()
        nc = _tuple3_add(ci, offset)
        if full_labels[nc] < 0 and new_labels[ci] == fill_val:
            full_labels[nc] = val
            idxs.extend(_find_neighbours(ci, new_labels.shape))


@jit(nopython=True)
def _update_labels(
    full_labels, new_labels, margin=(0, 0, 0), offset=(0, 0, 0), start_value=0
):
    for i in range(margin[0], new_labels.shape[0] - margin[0]):
        for j in range(margin[1], new_labels.shape[1] - margin[1]):
            for k in range(margin[2], new_labels.shape[2] - margin[2]):
                fl = full_labels[offset[0] + i, offset[1] + j, offset[2] + k]
                if fl == -1 and new_labels[i, j, k] >= 0:
                    _flood_fill(full_labels, new_labels, (i, j, k), offset, start_value)
                    start_value += 1
    return start_value
