import numpy as np

from fimpy.alignment.reg_from_skimage import register_translation
from scipy.ndimage.interpolation import shift
from skimage.filters import sobel
from scipy.ndimage.filters import gaussian_filter


def _samesize(source, target):
    """Either crops or expands the source image to have the same size as
    the target. Returns a view if a strictly smaller stack is requested

    :param source:
    :param target:
    :return:
    """
    ind_s = []
    ind_t = []

    bigger = False

    for dim_s, dim_t in zip(source.shape, target.shape):
        if dim_s < dim_t:
            ind_s.append(slice(None))
            start = (dim_t - dim_s) // 2
            ind_t.append(slice(start, start + dim_s))
        else:
            bigger = True
            ind_t.append(slice(None))
            start = (dim_s - dim_t) // 2
            ind_s.append(slice(start, start + dim_t))

    if bigger:
        new = np.zeros(target.shape, source.dtype)
        new[tuple(ind_t)] = source[tuple(ind_s)]
        return new
    else:
        return source[tuple(ind_s)]


def align_block_shift(stack, fft_ref, upsample_factor=10, maxshift=15):
    """Aligns an image stack using a reference (possibly from another stack)

    :param stack:
    :param reference:
    :param fft_ref:
    :param maxshift:
    :return:
    """
    shifts = np.empty((stack.shape[0], stack.ndim - 1))
    shifted = np.empty((stack.shape[0],) + fft_ref.shape, stack.dtype)

    for t in range(stack.shape[0]):
        to_align = stack[t]
        if to_align.shape != fft_ref.shape:
            to_align = _samesize(to_align, fft_ref)
        shifts[t, :] = register_translation(
            np.fft.fftn(to_align),
            fft_ref,
            upsample_factor=upsample_factor,
            space="fourier",
            return_error=False,
        )
        if np.any(np.abs(shifts[t, :]) > maxshift):
            shifted[t, ...] = to_align
        else:
            shifted[t, ...] = shift(to_align, -shifts[t])
    return shifted, shifts


def sobel_stack(stk, sigma=0):
    """Return a stack where each plane is filtered with a sobel filter
    after being gaussian filtered

    :param stk: the 3D image stack
    :param sigma: the standard deviation of the gaussian fileter
    :return: filtered 3D stack
    """
    # TODO implement in Fourier space
    if sigma == 0:
        return stk
    else:
        return np.stack([sobel(gaussian_filter(s, sigma)) for s in stk], 0)


def find_shifts_sobel(stack, fft_ref, sigma, upsample_factor=10):
    """Sobel-filters a 3D image and aligns it to reference which should
    already be in fourier space

    :param stack:
    :param fft_ref:
    :param sigma:
    :param upsample_factor:
    :return:
    """
    # # added these lines for accepting also 3D stack, without time:
    # if len(stack.shape) > 3:
    #     stack = np.mean(stack, 0)

    fft_mov = np.fft.fftn(sobel_stack(_samesize(np.mean(stack, 0), fft_ref), sigma))
    reg = register_translation(
        fft_mov,
        fft_ref,
        upsample_factor=upsample_factor,
        space="fourier",
        return_error=False,
    )

    return reg


def shift_stack(stack, stackframes, shifts, shift_times):
    shifted = np.empty_like(stack)
    for i, t in enumerate(stackframes):
        shift_amount = -np.array(
            [np.interp(t, shift_times, shifts[:, d]) for d in range(stack.ndim - 1)]
        )
        shifted[i, ...] = shift(stack[i, ...], shift_amount)
    return shifted
