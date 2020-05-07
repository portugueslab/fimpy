# from pathlib import Path
# from shutil import copy
# import numpy as np
# from fimpy.loading.volumetric import load_volumetric, fix_frameskip


# def test_lightsheet_pipeline(tmpdir):
#     dest_dir = Path(tmpdir.mkdir("ls"))
#     source_dir = Path(__file__).parent / "assets" / "ls"
#     for f in list(source_dir.glob("*")):
#         copy(f, dest_dir / f.name)
#
#     ds = load_volumetric(
#         source_dir, dest_dir, format="TIF", block_duration=20, verbose=True
#     )
#     assert ds.shape == (110, 30, 60, 100)
#     assert ds.shape_block == (40, 30, 60, 100)
#     assert len(list((dest_dir / "original").glob("*.h5"))) == 3
#
#     fixed_ds = fix_frameskip(
#         ds,
#         n_frames_process=30,
#         planes_skip_before=1,
#         planes_skip_after=3,
#         invert_z=True,
#         n_jobs=20,
#     )
#     assert fixed_ds.shape == (110, 26, 60, 100)
#     assert np.sum(fixed_ds[34, 19, :, :]) == 0
