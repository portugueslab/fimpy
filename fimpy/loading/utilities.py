import numpy as np
import imageio


def _stack_from_tif(file):
    reader = imageio.get_reader(file)
    img = reader.get_data(0)
    n_frames = reader.get_length()
    stack = np.full((n_frames, *img.shape), 0, dtype=np.uint16)
    for i in range(n_frames):
        try:
            stack[i, :, :] = reader.get_data(i)
        # fixes for files saved with errors
        except ValueError:
            return stack[0:i, :, :]
    return stack


def _imshape_from_tif(file):
    with imageio.get_reader(file) as reader:
        return reader.get_data(0).shape
