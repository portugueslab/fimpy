import numpy as np

import imageio
from fimpy.loading import StackSaver, loading_function
from pathlib import Path
import h5py

from split_dataset import SplitDataset


@loading_function
def load_2p(
    folder,
    folder_metadata=None,
    pattern="*tif",
    folder_dest=None,
    subtract=0,
    sort_key_funct=None,
    z_reversed=False,
    print_info=False,
    frames_per_plane=None,
):
    """
    :param folder:
    :param folder_dest:
    :return:
    """
    path = Path(folder)
    im_files = sorted(path.glob(pattern), key=sort_key_funct, reverse=z_reversed)
    f_type = im_files[0].name.split(".")[-1]

    if frames_per_plane is None:
        if folder_metadata is None:
            meta_path = next(path.glob("*be*"))
        else:
            meta_path = Path(folder_metadata)

        meta_files = sorted(meta_path.glob("*_metadata.json"))
        all_experiments = [Experiment(f) for f in meta_files]
        scope_config = all_experiments[-1]["imaging"]["microscope_config"]
        print("Microscope config: {}".format(scope_config))
        frames_per_plane = scope_config["frames_per_plane"]

    # get frames_per_stack from loaded stack shape, in case of loading from
    # preprocessed data:
    first_arr = read_stack(im_files[0])

    if f_type == "mat":  # If loading old .mat, they contain one plane
        stacks_per_plane = 1
    else:
        stacks_per_plane = frames_per_plane // first_arr.shape[0]

    print(
        "Stacks per plane: {} ({}/{})".format(
            stacks_per_plane, frames_per_plane, first_arr.shape[0]
        )
    )

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

    cont = StackSaver(
        destination=folder_dest, shape_full=shape_full, shape_block=shape_block
    )

    for i in range(0, len(im_files), stacks_per_plane):
        p = i // stacks_per_plane  # plane number
        if print_info:
            print("Loading plane {}...".format(p))
        stack = read_file_list(im_files[i : i + stacks_per_plane])
        stack[:, :, :] = np.maximum(stack, subtract) - subtract
        cont.save_block(p, stack[:, None, :, :])

    cont.save_metadata()
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
