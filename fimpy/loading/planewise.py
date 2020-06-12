import numpy as np

import imageio
from fimpy.loading import StackSaver, loading_function
from pathlib import Path
import h5py

from split_dataset import SplitDataset, EmptySplitDataset


@loading_function
def load_planewise(
    folder,
    stacks_per_plane=1,
    pattern="*tif",
    folder_dest=None,
    subtract=0,
    sort_key_funct=None,
    z_reversed=False,
    verbose=False,
):
    """
    :param folder:
    :param folder_dest:
    :return:
    """
    path = Path(folder)
    im_files = sorted(path.glob(pattern), key=sort_key_funct, reverse=z_reversed)

    # get frames_per_stack from loaded stack shape, in case of loading from
    # preprocessed data:
    first_arr = read_stack(im_files[0])
    frames_per_plane = first_arr.shape[0] * stacks_per_plane

    if np.mod(len(im_files), stacks_per_plane) != 0:
        raise FileNotFoundError(
            "Wrong number of files in folder! "
            "There should be a multiple of {} tiff files, "
            "however there are {}".format(stacks_per_plane, len(im_files))
        )

    shape_full = (
        frames_per_plane,
        len(im_files) // stacks_per_plane,
        first_arr.shape[1],
        first_arr.shape[2],
    )
    shape_block = (
        stacks_per_plane * first_arr.shape[0],
        1,
        first_arr.shape[1],
        first_arr.shape[2],
    )

    if folder_dest is None:
        folder_dest = path

    cont = EmptySplitDataset(name="original",
        root=folder_dest, shape_full=shape_full, shape_block=shape_block
    )

    for i in range(0, len(im_files), stacks_per_plane):
        p = i // stacks_per_plane  # plane number
        if verbose:
            print("Loading plane {}...".format(p))
        stack = read_file_list(im_files[i : i + stacks_per_plane])
        stack[:, :, :] = np.maximum(stack, subtract) - subtract
        cont.save_block_data(p, stack[:, np.newaxis, :, :])

    cont.finalize()
    return SplitDataset(folder_dest / "original")


def read_file_list(f_list):
    return np.concatenate([read_stack(f) for f in f_list], 0)


def read_stack(file):
    f_type = file.name.split(".")[-1]

    if f_type == "mat":
        with h5py.File(str(file), "r") as f:
            stack = f["data"].value
    elif f_type == "tif":
        stack = np.stack(imageio.mimread(str(file)), 0)

    return stack
