import numpy as np
from split_dataset import Blocks
from fimpy.roi_extraction.merging import merge_rois


def test_merging_rois():
    test_0 = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1],
            [0, 0, 0, -1, 1],
            [-1, -1, 2, 2, 2],
            [-1, -1, 2, -1, -1],
        ]
    )[None, :, :].astype(np.int32)
    test_1 = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1],
            [0, 0, 0, 1, 1],
            [-1, -1, 2, 2, 2],
            [-1, 2, 2, 0, 0],
        ]
    )[None, :, :].astype(np.int32)
    bl = Blocks((1, 5, 8), (1, 3, 3), padding=(0, 2, 2))
    block_arrays = np.array([[[test_0, test_1]]])
    labels = merge_rois(bl, block_arrays)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 0, 0, 0, 2, 2, 2, -1],
                    [0, 0, 0, 2, 2, 2, 3, 3],
                    [-1, -1, 1, 1, 1, 4, 4, 4],
                    [-1, -1, 1, -1, 4, 4, -1, -1],
                ]
            ]
        ),
    )
