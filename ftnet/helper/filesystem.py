"""Filesystem utility functions."""

from pathlib import Path


class Checkpoint:
    """Utility class to manage checkpoints and associated directories.

    Args:
        save_path (Path): Root path where checkpoints will be saved.
        test (bool, optional): Whether in test mode. Defaults to False.
    """

    def __init__(self, save_path: Path, test: bool = False):
        self.save_path = Path(save_path)
        if test:
            self.save_path /= "Evaluation"

        self.paths = {
            "save_dir": self.save_path,
            "logs": self.save_path / "logs",
        }
        if not test:
            self.paths["ckpt"] = self.save_path / "ckpt"

        # Create directories if they don't exist
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_path(self, subdir: str) -> Path:
        """Retrieve the path for a specific subdirectory under the checkpoint
        directory.

        Args:
            subdir (str): Subdirectory name.

        Returns:
            Path: Path to the requested subdirectory.
        """
        if subdir in self.paths:
            return self.paths[subdir]

        temp = self.save_path / subdir
        temp.mkdir(parents=True, exist_ok=True)
        self.paths[subdir] = temp
        return temp
