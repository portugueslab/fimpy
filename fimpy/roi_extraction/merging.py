from numba import jit
from fimpy.roi_extraction.corr_algorithms import _update_labels
from split_dataset import Blocks
import numpy as np
from itertools import product


def merge_rois(blocks: Blocks, block_arrays: np.ndarray):
    """ Merges rois extracted from different dataset blocks with padding

    :param blocks:
    :param block_arrays:
    :return:
    """
    new_rois = np.full(blocks.shape_full, -1, np.int32)
    counter = 0
    margin = tuple(p // 2 for p in blocks.padding)
    for (i, j, k) in product(*(range(x) for x in blocks.block_starts.shape[:-1])):
        c_offset = tuple(blocks.block_starts[i, j, k])
        counter = _update_labels(
            new_rois,
            block_arrays[i, j, k],
            margin=margin,
            offset=c_offset,
            start_value=counter,
        )
    return new_rois
