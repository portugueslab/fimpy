import numpy as np
from scipy.ndimage.interpolation import shift
from fimpy.alignment.volume import sobel_stack
from skimage.registration import phase_cross_correlation


def align_single_planes_sobel(
    stack,
    fft_ref=None,
    prefilter_sigma=3.3,
    upsample_factor=10,
    maxshift=15,
    offset=[0, 0],
):
    """Aligns planes in time, z dimension is space

    :param stack: the input video, dimensions [t, z, y, x]
    :param reference : the reference image, if not present, align to the mean
        of all, or take_first frames
    :param take_first: the number of frames to take to generate the reference
    :param algorithm: ecc for translation ecc-euc for translation and rotation
        or ecc-af for affine transforms or thunder
    :return: aligned video
    """
    offset = np.array(offset)

    shifts = np.empty((stack.shape[0], stack.shape[1], 2))
    shifted = np.empty((stack.shape[0],) + fft_ref.shape, stack.dtype)

    for i in range(stack.shape[1]):
        to_fix = sobel_stack(stack[:, i, :, :], prefilter_sigma)
        l = []
        for t in range(to_fix.shape[0]):
            shifts[t, i, :] = (
                phase_cross_correlation(
                    np.fft.fftn(to_fix[t]),
                    fft_ref[i],
                    space="fourier",
                    upsample_factor=upsample_factor,
                    return_error=False,
                )
                + offset
            )

            if np.any(np.abs(shifts[t, i, :] - offset) > maxshift):
                l.append(t)
                shifted[t, i, :, :] = 0  # to_fix[t, :, :]
            else:
                shifted[t, i, :, :] = shift(
                    stack[t, i, :, :], -shifts[t, i, :], order=1
                )
        if len(l) > 0:
            print("got problems in frames {}".format(l))
    return shifted, shifts
