import unittest

import numpy as np

from fimpy.roi_extraction.local_correlations import grow_rois, correlation_map

from fimpy.roi_extraction.anatomical import extract_traces_around_points


def test_positive_corr():
    video = np.array(
        [
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
                [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
            ],
            [
                [[1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]],
                [[2, 2, 2, 2], [0, 2, 2, 2], [0, 0, 2, 0]],
                [[3, 3, 3, 3], [0, 3, 3, 3], [0, 0, 3, 0]],
                [[2, 2, 2, 2], [0, 2, 2, 2], [0, 0, 2, 0]],
            ],
            [
                [[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]],
                [[2, 0, 2, 1], [0, 2, 2, 2], [0, 0, 2, 0]],
                [[3, 0, 3, 1], [0, 3, 3, 3], [0, 0, 3, 0]],
                [[2, 0, 2, 1], [0, 2, 1, 2], [0, 0, 2, 0]],
            ],
        ]
    )

    video = np.swapaxes(video, 0, 1)
    cm = correlation_map(video, window_size=(1, 3, 3))

    expected_result = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.375, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.10660036, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )

    np.testing.assert_allclose(cm, expected_result, atol=2 / 2 ** 15)


def test_big_vals():
    video = (
        np.array(
            [
                [[1, 1, 1], [0, 1, 1], [0, 0, 0]],
                [[255, 255, 255], [0, 255, 255], [0, 0, 0]],
                [[3, 3, 3], [0, 3, 3], [0, 0, 0]],
                [[255, 255, 255], [0, 255, 255], [0, 0, 0]],
            ]
        )
        * 2 ** 4
    )

    cm = correlation_map(video[:, None, :, :], window_size=(1, 3, 3))

    result = np.array([[0, 0, 0], [0, 0.375, 0], [0, 0, 0]])[None, :, :]

    assert np.all(np.isclose(cm, result, atol=2 / 2 ** 15))


def test_simple_flood_case():
    video = np.array(
        [
            [
                [[1, 1, 1], [-1, 1, 1], [-1, -1, -1]],
                [[2, 2, 2], [-2, 2, 2], [-2, -2, -2]],
                [[3, 3, 3], [-3, 3, 3], [-3, -3, -3]],
                [[2, 2, 2], [-2, 2, 2], [-2, -2, -2]],
            ],
            [
                [[1, 1, 1], [0, 1, 1], [0, 0, 0]],
                [[2, 2, 2], [0, 2, 2], [0, 0, 0]],
                [[3, 3, 3], [0, 3, 3], [0, 0, 0]],
                [[2, 2, 2], [0, 2, 2], [0, 0, 0]],
            ],
            [
                [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                [[0, 2, 1], [0, 2, 2], [0, 0, 0]],
                [[0, 3, 1], [0, 3, 3], [0, 0, 0]],
                [[0, 2, 1], [0, 2, 1], [0, 0, 0]],
            ],
        ]
    )

    video = np.swapaxes(video, 0, 1)

    cm = correlation_map(video)

    rois = grow_rois(
        video,
        cm,
        init_corr_thresh=0.9,
        final_corr_threshold=0.95,
        max_investigate=2,
        max_labels=4,
    )
    roi_result = np.array(
        [
            [[1, -1, -1], [-1, 0, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]],
        ]
    )

    assert np.all(rois == roi_result)


def test_point_extraction():
    test_stack = np.stack([np.zeros((7,) * 3), np.ones((7,) * 3)])
    test_points = np.array([[1, 1, 1], [3, 3, 3]])
    test_kernel = (0.1, 0.1, 0.1)
    result = extract_traces_around_points(test_stack, test_points)
    desired = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert np.allclose(result["traces"], desired)


if __name__ == "__main__":
    unittest.main()
