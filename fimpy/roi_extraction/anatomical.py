from numba import jit
import math
import numpy as np


def extract_traces_around_points(stack, points, **kwargs) -> dict:
    return dict(traces=_extract_traces_around_points_jit(stack, points, **kwargs))


@jit(nopython=True)
def _extract_traces_around_points_jit(
    stack, points, kernel=(0.5, 1, 1), kernel_mult=2, minimal_weight=10
):
    """ Extracts traces around points with a Gaussian profile

    :param stack: the stack from which the traces are to be extracted
    :param points: the points around which to extract the traces, shape
        [#points, 3]
    :param kernel: the standard deviations for the Gaussian used for weighting
    :param kernel_mult: how many standard deviations to take into account
    :return: an array with traces
    """
    point_sums = np.zeros((points.shape[0], stack.shape[0]), dtype=np.float64)
    for ip in range(points.shape[0]):
        point = points[ip]

        zs = min(
            max(int(math.floor(point[0] - kernel[0] * kernel_mult)), 0),
            stack.shape[1] - 1,
        )
        ze = min(
            max(int(math.ceil(point[0] + kernel[0] * kernel_mult)), 0),
            stack.shape[1] - 1,
        )
        ys = min(
            max(int(math.floor(point[1] - kernel[1] * kernel_mult)), 0),
            stack.shape[2] - 1,
        )
        ye = min(
            max(int(math.ceil(point[1] + kernel[1] * kernel_mult)), 0),
            stack.shape[2] - 1,
        )
        xs = min(
            max(int(math.floor(point[2] - kernel[2] * kernel_mult)), 0),
            stack.shape[3] - 1,
        )
        xe = min(
            max(int(math.ceil(point[2] + kernel[2] * kernel_mult)), 0),
            stack.shape[3] - 1,
        )

        # iterate through the window in 3 dimensions
        w_sum = 0.0
        for z in range(zs, ze + 1):
            for y in range(ys, ye + 1):
                for x in range(xs, xe + 1):

                    # calculate the weight from the Gaussian
                    w = (
                        math.exp(-(z - point[0]) ** 2 / (2 * kernel[0]))
                        * math.exp(-(y - point[1]) ** 2 / (2 * kernel[1]))
                        * math.exp(-(x - point[2]) ** 2 / (2 * kernel[2]))
                    )
                    w_sum += w
                    # accumulate this to the corresponding point
                    point_sums[ip, :] += stack[:, z, y, x] * w
        if w_sum <= minimal_weight:
            point_sums[ip, :] = 0
        else:
            point_sums[ip, :] /= w_sum

    return point_sums


def paint_anatomy(stack: np.ndarray, points, color=(230, 40, 0), **kwargs):
    """ Paint the anatomical ROI extraction colors over the anatomy

    :param stack: the stack in which to paint the extraction kernels
    :param points: coordinates of the kernel centers
    :param color: RGB (0-255) color of the kernels
    :param kwargs: the kernel arguments: kernel as as set of widths and kernel_mult
    :return:
    """
    if stack.dtype != np.uint8:
        stack = (stack * 255).astype(np.uint8)

    if stack.ndim < 4:
        stack = np.tile(stack[:, :, :, None], (1, 1, 1, 3))
    color = np.array(color)
    return _paint_anatomy_jit(stack, points, color, **kwargs)


@jit(nopython=True)
def _paint_anatomy_jit(stack, points, color, kernel=(0.5, 1, 1), kernel_mult=2):
    """ Extracts traces around points with a Gaussian profile

    :param stack: the stack from which the traces are to be extracted
    :param points: the points around which to extract the traces, shape
        [#points, 3]
    :param kernel: the standard deviations for the Gaussian used for weighting
    :param kernel_mult: how many standard deviations to take into account
    :return: an array with traces
    """
    for ip in range(points.shape[0]):
        point = points[ip]

        zs = min(
            max(int(math.floor(point[0] - kernel[0] * kernel_mult)), 0),
            stack.shape[0] - 1,
        )
        ze = min(
            max(int(math.ceil(point[0] + kernel[0] * kernel_mult)), 0),
            stack.shape[0] - 1,
        )
        ys = min(
            max(int(math.floor(point[1] - kernel[1] * kernel_mult)), 0),
            stack.shape[1] - 1,
        )
        ye = min(
            max(int(math.ceil(point[1] + kernel[1] * kernel_mult)), 0),
            stack.shape[1] - 1,
        )
        xs = min(
            max(int(math.floor(point[2] - kernel[2] * kernel_mult)), 0),
            stack.shape[2] - 1,
        )
        xe = min(
            max(int(math.ceil(point[2] + kernel[2] * kernel_mult)), 0),
            stack.shape[2] - 1,
        )

        # iterate throught the window in 3 dimensions
        for z in range(zs, ze + 1):
            for y in range(ys, ye + 1):
                for x in range(xs, xe + 1):

                    # calculate the weight from the Gaussian
                    w = (
                        math.exp(-(z - point[0]) ** 2 / (2 * kernel[0]))
                        * math.exp(-(y - point[1]) ** 2 / (2 * kernel[1]))
                        * math.exp(-(x - point[2]) ** 2 / (2 * kernel[2]))
                    )

                    stack[z, y, x, :] = stack[z, y, x, :] * (1 - w) + color * w
    return stack


def extract_traces_from_roi_array(stack, pointsdata, twop=False):
    """ Extracts traces from rois defined as a list of points with assigned identities

    :param stack:
    :param pointsdata:
    :return:
    """
    coords, areas, traces = _extract_trace_from_roi_array_jit(
        stack,
        pointsdata["roi_locations"],
        pointsdata["roi_identities"].astype(np.int64),
    )

    return dict(coords=coords, areas=areas, traces=traces)


@jit(nopython=True, parallel=True)
def _extract_trace_from_roi_array_jit(stack, roi_array, roi_identities):
    """ Extract traces from an array containging ROI locations and identities

    :param stack:
    :param roi_array:
    :param roi_identities:
    :return:
    """
    visited = np.zeros(stack.shape[1:], dtype=np.uint8)
    n_rois = roi_identities[-1]
    traces = np.zeros((n_rois, stack.shape[0]))
    coords = np.zeros((n_rois, 3))
    sums = np.zeros(n_rois)
    for k in range(len(roi_identities)):
        point_fl = roi_array[k]
        roi_id = roi_identities[k] - 1
        point = (
            int(round(point_fl[0])),
            int(round(point_fl[1])),
            int(round(point_fl[2])),
        )
        if (
            (0 <= point[0] < visited.shape[0])
            and (0 <= point[1] < visited.shape[1])
            and (0 <= point[2] < visited.shape[2])
        ):
            if visited[point] == 0:
                visited[point] = 1
                for t in range(stack.shape[0]):
                    traces[roi_id, t] += stack[(t,) + point]
                for i in range(3):
                    coords[roi_id, i] += point[i]
                sums[roi_id] += 1
    for i in range(traces.shape[0]):
        if sums[i] > 0:
            traces[i, :] /= sums[i]
            coords[i, :] /= sums[i]
    return coords, sums, traces


@jit(nopython=True)
def get_ROI_coords_areas_traces_3D_planewise(stack, rois: np.ndarray) -> dict:
    """ A function to efficiently extract ROI data, 3D ROIs

    :param stack: imaging stack
    :param rois: image where each ROI is labeled by the same integer
    :param max_rois: number of ROIs per stack
    :return: coords, areas, traces
    """
    n_rois = np.max(rois + 1)
    n_time = stack.shape[0]
    n_planes = stack.shape[1]
    traces = np.zeros((n_rois, n_planes, n_time), dtype=np.float32)
    coords = np.zeros((n_rois, len(rois.shape)))
    areas = np.zeros((n_rois, n_planes), np.int32)
    for i in range(rois.shape[0]):
        for j in range(rois.shape[1]):
            for k in range(rois.shape[2]):
                roi_id = rois[i, j, k]
                if roi_id > -1:
                    areas[roi_id] += 1
                    coords[roi_id, 0] += i
                    coords[roi_id, 1] += j
                    coords[roi_id, 2] += k
                    traces[roi_id, i, :] += stack[:, i, j, k]

    for i in range(n_rois):
        coords[i, :] /= areas[i]
        for z in range(n_planes):
            traces[i, z, :] /= areas[i, z]

    return coords, areas, traces
