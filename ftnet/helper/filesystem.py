"""Filesystem utility functions."""

import errno
import os
from pathlib import Path

__all__ = ["checkpoint", "makedirs"]


class checkpoint:
    def __init__(self, save_path: Path, test=False):
        self.save_path = save_path
        if test:
            self.save_path = Path(self.save_path) / "Evaluation"

        self.paths = {
            "save_dir": Path(self.save_path),
            "logs": Path(self.save_path) / "logs",
        }
        if not test:
            self.paths["ckpt"] = Path(self.save_path) / "ckpt"

        for _, val in self.paths.items():
            val.mkdir(parents=True, exist_ok=True)

    def get_path(self, subdir: str) -> Path:
        if subdir in self.paths:
            return self.paths[subdir]

        temp = Path(self.save_path) / subdir
        self.paths[subdir] = temp
        temp.mkdir(parents=True, exist_ok=True)
        return temp


def makedirs(path: str):
    """Create directory recursively if not exists. Similar to `makedir -p`, you
    can skip checking existence before this function.

    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path, mode=0o770, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def try_import(package, message=None):
    """Try import specified package, with custom message support.

    Parameters
    ----------
    package : str
        The name of the targeting package.
    message : str, default is None
        If not None, this function will raise customized error message when import error is found.
    Returns
    -------
    module if found, raise ImportError otherwise
    """
    try:
        return __import__(package)
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)


def try_import_cv2():
    """Try import cv2 at runtime.

    Returns
    -------
    cv2 module if found. Raise ImportError otherwise
    """
    msg = "cv2 is required, you can install by package manager, e.g. 'apt-get', \
        or `pip install opencv-python --user` (note that this is unofficial PYPI package)."
    return try_import("cv2", msg)


def import_try_install(package, extern_url=None):
    """Try import the specified package. If the package not installed, try use
    pip to install and import if success.

    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.
    Returns
    -------
    <class 'Module'>
        The imported python module.
    """
    try:
        return __import__(package)
    except ImportError:
        try:
            from pip import main as pipmain
        except ImportError:
            from pip._internal import main as pipmain

        # trying to install package
        url = package if extern_url is None else extern_url
        pipmain(["install", "--user", url])  # will raise SystemExit Error if fails

        # trying to load again
        try:
            return __import__(package)
        except ImportError:
            import site
            import sys

            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            return __import__(package)
    return __import__(package)