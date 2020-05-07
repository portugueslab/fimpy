import numpy as np
from fimpy.roi_extraction.corr_algorithms import _correlation_map_jit, _jit_flood
import itertools
from fimpy.utilities import to_4d, to_3d
from lightparam import Param


def correlation_map(video: np.ndarray, window_size=(1, 3, 3)) -> np.ndarray:
    """ Calculate the correlation map of a video. Tiny wrapper around the
    _correlation_map_jit, to deal with the non numba-izable stuff.
    :param video: the 4D stack [t, z, x, y]
    :param window_size: tuple with full window size along each axis
    :return: the 3D correlation map
    """

    video = to_4d(video)  # ensures video is 4D

    wrs = tuple([ws // 2 for ws in window_size])  # calculate half-sizes
    ranges = [range(-wr, wr + 1) for wr in wrs]

    # produce tuples of neighbours:
    neighbours = tuple(itertools.product(*ranges))
    n_neighbours = len(neighbours) // 2

    return _correlation_map_jit(
        video,
        wrs,
        tuple(neighbours[:n_neighbours]),
        tuple(neighbours[n_neighbours + 1 :]),
    )


def grow_rois(
    stack,
    corr_map,
    init_corr_thresh: Param(0.15, (0.01, 0.95)) = 0.15,
    corr_thresh_inc_dist: Param(1.0, (0.5, 15)) = 1,
    final_corr_threshold: Param(0.0, (0, 0.95)) = 0,
    max_radius: Param(10, (2, 100)) = 10,
    min_area: Param(1, (1, 50)) = 1,
    max_labels: Param(100, (10, 20000)) = 100,
    max_investigate: Param(2000, (100, 30000)) = 2000,
    across_planes: Param(True) = True,
    voxel_size=(1, 1, 1),
) -> np.ndarray:
    """ Find ROIs by taking local maxima of correlation maps and growing ROIs
     around them.
     !! NOTE that the correlation map is used ONLY to sort the indexes in order
     of priority of investigation; the correlation threshold is then applied to
     the correlation of the individual pixel traces, of which the correlation
     map values are a good but not perfect approximation.

    :param corr_map: input correlation map
    :param init_corr_thresh: the threshold of correlation that gets added to
        the roi, grows linearly until corr_thresh_inc_dist
    :param corr_thresh_inc_dist: the distance until which the correlation
        threshold increases, in voxels if voxel_size is not specified, otherwise
        in um
     :param final_corr_threshold: the correlation threshold reached over
        corr_threshold_steps
    :param max_radius: the maximum radius of a ROIs, so that they are
        roughly circular
    :param min_area: the minimum area of a ROI
    :param max_labels: the maximal number of ROIs to search for
        (if not across planes, in one plane)
    :param max_investigate: the maximum number of potential ROIs to investigate
        per slice or volume part (as small ROIs are discarded)

    :param across_planes: True to grow ROIs across planes
    :param voxel_size: (optional, size of voxels in mm)


    :return: stack ROIs labeled
    """

    # if we get the video or the correlation map of the wrong shape,
    #  expand it so that the algorithm still works
    stack = to_4d(stack)
    corr_map = to_3d(corr_map)

    return _jit_flood(
        stack,
        corr_map,
        init_corr_thresh,
        max_labels,
        final_corr_threshold,
        max_radius,
        corr_thresh_inc_dist,
        max_investigate,
        min_area,
        across_planes,
        voxel_size,
    )
