import json
import flammkuchen as fl
from pathlib import Path

from split_dataset import Blocks


def loading_function(f):
    f.loading_function = True
    return f


class StackSaver(Blocks):
    """
    """

    def __init__(
        self, *args, destination=None, name="original", verbose=False, **kwargs
    ):
        """
        :param destination:
        :param n_planes:
        :param time_block_duration:
        :param metadata:
        """
        super().__init__(*args, **kwargs)
        self.path = Path(destination) / name
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.files = []
        self.verbose = verbose

    def save_block(self, i, array_to_write):
        """
        :param i:
        :param array_to_write:
        :return:
        """
        if not array_to_write.shape == self.shape_block:
            print("Array size smaller than block dim")
        fname = "{:03d}.h5".format(i)
        if self.verbose:
            print("Saving {}".format(str(self.path / fname)))
        self.files.append(fname)
        to_save = dict(
            stack_4D=array_to_write,
            position=self.block_starts[self.linear_to_cartesian(i) + (slice(None),)],
        )
        fl.save(str(self.path / fname), to_save, compression="blosc")

    def save_metadata(self):
        """ Save stack metadata in json file
        :return:
        """
        stack_metadata = self.serialize()
        stack_metadata.update(dict(files=self.files))
        with open(str(self.path / "stack_metadata.json"), "w") as f:
            json.dump(stack_metadata, f, sort_keys=True)
