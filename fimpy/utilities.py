from numba import jit
import numpy as np
import math

@jit(nopython=True)
def fast_pearson(x, y):
    """ Calculate correlation between two data series.
    :param x: first array
    :param y: second array
    :return: pearson correlation
    """
    n = len(x)
    s_xy = 0.0
    s_x = 0.0
    s_y = 0.0
    s_x2 = 0.0
    s_y2 = 0.0
    for i in range(n):
        s_xy += x[i] * y[i]
        s_x += x[i]
        s_y += y[i]
        s_x2 += x[i] ** 2
        s_y2 += y[i] ** 2
    denominator = math.sqrt((s_x2 - (s_x ** 2) / n) * (s_y2 - (s_y ** 2) / n))
    if denominator == 0:
        return 0
    return (s_xy - s_x * s_y / n) / denominator


def to_3d(stack):
    n = 3
    if len(stack.shape) == n - 1:
        return stack[None, ...]
    elif len(stack.shape) == n:
        return stack
    else:
        raise Exception("Input array has to be {} or {} dimensional".format(n, n - 1))


def to_4d(stack):
    n = 4
    if len(stack.shape) == n - 1:
        return stack[:, None, :, :]
    elif len(stack.shape) == n:
        return stack
    else:
        raise Exception("Input array has to be {} or {} dimensional".format(n, n - 1))


def nanzscore(array):
    return (array - np.nanmean(array)) / np.nanstd(array)
