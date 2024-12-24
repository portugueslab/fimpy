from pathlib import Path
import imageio
import numpy as np

from split_dataset import EmptySplitDataset
from fimpy.loading import loading_function
from fimpy.pipeline.common import run_in_blocks
from fimpy.loading.utilities import _stack_from_tif, _imshape_from_tif


@loading_function
def load_volumetric(
    data_dir,
    n_planes,
    filesort_key=None,
    output_dir=None,
    block_duration=300,
    resolution=(1, 1, 1, 1),
    verbose=True,
):
    """
    Load a tiff micromanager file into a split h5 dataset.
    :param data_dir:
    :param output_dir:
    :param fish_id:
    :param session_id:
    :return:
    """

    if output_dir is None:
        output_dir = data_dir

    data_dir, output_dir = Path(data_dir), Path(output_dir)

    # find all files:
    im_files = sorted(list(data_dir.glob("*.tif")), key=filesort_key)
    if len(im_files) == 0:
        raise Exception("The selected folder contains no tiff files!")

    im_shape = _imshape_from_tif(im_files[0])  # shape of a single frame

    # Not knowing the shape full at this point, we will just make an empty
    # container with full_shape equal to block shape. This will be corrected
    # at the end by the StackContainerLs object before saving stack metadata.
    loaded_ds = StackContainerLs(
        output_dir,
        shape_full=(block_duration, n_planes) + im_shape,
        shape_block=(block_duration, n_planes) + im_shape,
        resolution=resolution,
    )

    # Loop over files, read and pour reshaped array:
    for tf in im_files:
        try:
            if verbose:
                print("Loading ", tf)
            # Read and pour:

        except RuntimeError as e:
            print("Failed reading {}".format(tf))
            print(str(e))

    return loaded_ds.finalize()


@loading_function
def load_volumetric_new_reader(
    data_dir,
    n_planes,
    filesort_key=None,
    output_dir=None,
    block_duration=300,
    resolution=(1, 1, 1, 1),
    verbose=True,
):
    """
    Load a tiff micromanager file into a split h5 dataset.
    :param data_dir:
    :param output_dir:
    :param fish_id:
    :param session_id:
    :return:
    """

    if output_dir is None:
        output_dir = data_dir

    data_dir, output_dir = Path(data_dir), Path(output_dir)

    # find all files:
    im_files = sorted(list(data_dir.glob("*.tif")), key=filesort_key)
    if len(im_files) == 0:
        raise Exception("The selected folder contains no tiff files!")

    im_shape = _imshape_from_tif(im_files[0])  # shape of a single frame
    if len(im_shape) > 2:
        im_shape = im_shape[1:]

    # Not knowing the shape full at this point, we will just make an empty
    # container with full_shape equal to block shape. This will be corrected
    # at the end by the StackContainerLs object before saving stack metadata.

    loaded_ds = StackContainerLs(
        output_dir,
        shape_full=(block_duration, n_planes) + im_shape,
        shape_block=(block_duration, n_planes) + im_shape,
        resolution=resolution,
    )

    tf = im_files[0]
    try:
        if verbose:
            print("Loading ", tf)
        # Read and pour:
        data = _stack_from_tif(str(tf))
        print(data.shape)
        if len(data.shape) > 3:
            data = data[0, ...]
        loaded_ds.pour(data)

    except RuntimeError as e:
        print("Failed reading {}".format(tf))
        print(str(e))

    return loaded_ds.finalize()


class StackContainerLs(EmptySplitDataset):
    """Class to conveniently pour 3D files from MicroManager
    into H5SplitDatasets
    """

    def __init__(self, root, **kwargs):
        """
        :param destination:
        :param n_planes:
        :param time_block_duration:
        :param metadata:
        """
        NAME = "original"  # default name for imported data
        super().__init__(root, NAME, **kwargs)
        self.time_block_duration = self.shape_block[0]
        self.n_planes = self.shape_block[1]
        self.arr_write = None
        self.total_duration = 0
        self.i_plane = 0
        self.i_frame = 0
        self.block_count = 0
        self.i_frame_absolute = 0
        self.files = []
        self.plane_shape = None

    def pour(self, data):
        """A recursive procedure which repeats until all of the input data
        is poured in blocks
        """
        if self.plane_shape is None:
            self.plane_shape = data.shape[1:]
        if self.arr_write is None:
            self.arr_write = np.empty(
                (self.time_block_duration, self.n_planes) + data.shape[1:],
                dtype=data.dtype,
            )
        if data.shape[0] == 0:
            return True

        if data.shape[0] < self.n_planes:
            self.arr_write[
                self.i_frame, self.i_plane : self.i_plane + data.shape[0], :, :
            ] = data[:]
            self.i_plane += data.shape[0]
            return True

        i_data = 0
        if self.i_plane != 0:
            self.arr_write[self.i_frame, self.i_plane : self.n_planes, :, :] = data[
                : self.n_planes - self.i_plane, :, :
            ]
            i_data = self.n_planes - self.i_plane
            self.i_plane = 0
            self.i_frame += 1

        n_to_pour = min(
            (self.time_block_duration - self.i_frame) * self.n_planes,
            self.n_planes * ((data.shape[0] - i_data) // self.n_planes),
        )
        n_frames_pour = n_to_pour // self.n_planes

        self.arr_write[self.i_frame : self.i_frame + n_frames_pour, :, :] = data[
            i_data : i_data + n_to_pour, :, :
        ].reshape(n_frames_pour, self.n_planes, data.shape[1], data.shape[2])

        self.i_frame += n_frames_pour

        if self.i_frame == self.time_block_duration:
            self.save_block()
            self.i_frame_absolute += self.time_block_duration
            self.i_frame = 0

        i_data += n_to_pour
        self.pour(data[i_data:])

    def save_block(self):
        self.block_count += 1
        self.total_duration += self.i_frame
        self.save_block_data(
            self.block_count, self.arr_write[: self.i_frame, :, :], verbose=True
        )

    def finalize(self):
        if self.i_frame != 0:
            self.save_block()

        # Correct the shape_full variable with the final number of volumes
        self.shape_full = (self.total_duration, self.n_planes) + self.plane_shape
        return super().finalize()


def fix_frameskip(
    ds,
    n_frames_process=None,
    planes_skip_before=1,
    planes_skip_after=3,
    n_jobs=20,
    invert_z=True,
):
    """Corrects ligthsheet stacks where frame skips mess up the z-order of planes.
    It applies the _fix_frameskip function using run_in_blocks.
    :param ds:  split dataseet to be fixed
    :param n_frames_process: number of frames for each task
    :param init_vol_skip: first 2 volumes will be skipped as they usually are not aligned
    :param plane_skip_before: number of planes to skip before objective drops
    :param plane_skip_after: number of planes to skip after ibjective drops
    :param n_jobs: number of jobs for the task
    :param invert_z: invert order of planes to follow convention of dorso-ventral
    :return:
    """

    # Heuristics to find reasonable splitting if none provided:
    if n_frames_process is None:
        n_frames_process = ds.shape[0] // 100

    # Calculate final shape:
    output_shape_full = (
        (ds.shape[0],)
        + (ds.shape[1] - planes_skip_before - planes_skip_after,)
        + ds.shape[2:]
    )
    return run_in_blocks(
        _fix_frameskip,
        ds,
        output_name="fix_frameskip",
        output_shape_full=output_shape_full,
        process_shape_block=(n_frames_process, *ds.shape[1:]),
        planes_skip_before=planes_skip_before,
        planes_skip_after=planes_skip_after,
        invert_z=invert_z,
        n_jobs=n_jobs,
    )


def _align_frame_to_ref(ref_frame, input_frame):
    """Function to fix a volume frame where a plane was skipped.
    It uses a reference to reconstruct the optimal plane-wise
    correspondance and ir needed leaves empty the missing plane.
    """
    fixed_frame = np.zeros(ref_frame.shape, dtype=ref_frame.dtype)
    for i_frame in range(input_frame.shape[0]):
        diff = np.abs(
            ref_frame.astype(np.float) - input_frame[i_frame, :, :].astype(np.float)
        ).sum((1, 2))
        new_i = np.argmin(diff)
        fixed_frame[new_i, :, :] = input_frame[i_frame, :, :]

    return fixed_frame


def _fix_frameskip(video, planes_skip_before=1, planes_skip_after=3, invert_z=True):
    """Corrects ligthsheet stacks where frame skips mess up the z-order of planes.
    :param video:  4D stack to be fixed
    :param plane_skip_before: number of planes to skip before objective drops
    :param plane_skip_after: number of planes to skip after ibjective drops
    :return:
    """
    # Initialize empty new stack:
    corrected_video = np.zeros(video.shape, dtype=video.dtype)
    n_planes = video.shape[1]
    n_frames = video.shape[0]

    # Store all positions with potentially bad frames (hopefully they are rare)
    all_skips = []
    prev_i_ins = None
    for i_frame in range(n_frames):
        stack = video[i_frame, :, :, :]

        # Use inter-plane differences to find point of objective dropping:
        differences = np.sum(
            np.diff(np.concatenate([stack[-1:, :, :], stack], axis=0), axis=0) ** 2,
            (1, 2),
        )
        i_insert = np.argmax(differences)
        # Fill in 2 times the corrected_video stack:
        corrected_video[i_frame, 0 : n_planes - i_insert, :, :] = stack[
            i_insert:n_planes, :, :
        ]
        corrected_video[i_frame, n_planes - i_insert : n_planes, :, :] = stack[
            0:i_insert
        ]

        if prev_i_ins is not None and prev_i_ins != i_insert:
            all_skips.append(i_frame)
        prev_i_ins = i_insert

    # Fix one by one suspected volumes
    # (currently inefficient but should happen only 1-10 times per exp):

    for i_skip in all_skips:
        if i_skip > 2:
            corrected_video[i_skip - 1, :, :, :] = _align_frame_to_ref(
                corrected_video[i_skip - 2, :, :, :],
                corrected_video[i_skip - 1, :, :, :],
            )

        corrected_video[i_skip, :, :, :] = _align_frame_to_ref(
            corrected_video[i_skip - 1, :, :, :], corrected_video[i_skip, :, :, :]
        )

    # Remove first and last frames (around objective dropping):
    corrected_video = corrected_video[
        :, planes_skip_after : n_planes - planes_skip_before, :, :
    ]

    # Flip over z if required:
    if invert_z:
        corrected_video = np.flip(corrected_video, axis=1)

    return corrected_video
