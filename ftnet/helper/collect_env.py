import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from tabulate import tabulate


def collect_torch_env() -> str:
    """Collects and returns environment information related to PyTorch.

    Returns:
        str: Environment information related to PyTorch.
    """
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # Compatible with older versions of PyTorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module() -> tuple:
    """Retrieves and returns environment module information.

    Returns:
        tuple: Variable name and its corresponding environment value.
    """
    var_name = "FTNET_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME: Path, so_file: Path) -> str:
    """Detects and returns compute compatibility of a CUDA-enabled device.

    Args:
        CUDA_HOME (Path): Path to CUDA installation directory.
        so_file (Path): Path to the shared object (SO) file.

    Returns:
        str: Compute compatibility information.
    """
    try:
        cuobjdump = CUDA_HOME / "bin" / "cuobjdump"
        if cuobjdump.exists():
            output = subprocess.check_output(f"'{cuobjdump}' --list-elf '{so_file}'", shell=True)
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)

        return f"{so_file}; cannot find cuobjdump"
    except Exception:
        # Unhandled failure
        return str(so_file)


def collect_env_info() -> str:
    """Collects and returns comprehensive environment information.

    Returns:
        str: Comprehensive environment information.
    """
    has_gpu = torch.cuda.is_available()  # True for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if getattr(torch.version, "hip", None) is not None and ROCM_HOME is not None:
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # Check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))
    data.append(get_env_module())
    data.append(("PyTorch", f"{torch_version} @{os.path.dirname(torch.__file__)}"))
    data.append(("PyTorch debug build", torch.version.debug))
    try:
        data.append(("torch._C._GLIBCXX_USE_CXX11_ABI", torch._C._GLIBCXX_USE_CXX11_ABI))
    except Exception:
        pass

    has_gpu_text = "Yes" if has_gpu else "No: torch.cuda.is_available() == False"
    data.append(("GPU available", has_gpu_text))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join(str(x) for x in torch.cuda.get_device_capability(k))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not (ROCM_HOME and Path(ROCM_HOME).is_dir()) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            try:
                from torch.utils.collect_env import get_nvidia_driver_version
                from torch.utils.collect_env import run as _run

                data.append(("Driver version", get_nvidia_driver_version(_run)))
            except Exception:
                pass
            msg = " - invalid!" if not (CUDA_HOME and Path(CUDA_HOME).is_dir()) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(
            ("torchvision", f"{torchvision.__version__} @{os.path.dirname(torchvision.__file__)}")
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except (ImportError, AttributeError):
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import fvcore

        data.append(("fvcore", fvcore.__version__))
    except (ImportError, AttributeError):
        pass

    try:
        import iopath

        data.append(("iopath", iopath.__version__))
    except (ImportError, AttributeError):
        pass

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except (ImportError, AttributeError):
        data.append(("cv2", "Not found"))

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def test_nccl_ops() -> None:
    """Tests NCCL operations for CUDA devices."""
    num_gpu = torch.cuda.device_count()
    if os.access("/tmp", os.W_OK):
        import torch.multiprocessing as mp

        dist_url = "file:///tmp/nccl_tmp_file"
        print("Testing NCCL connectivity ... this should not hang.")
        mp.spawn(_test_nccl_worker, nprocs=num_gpu, args=(num_gpu, dist_url), daemon=False)
        print("NCCL succeeded.")


def _test_nccl_worker(rank: int, num_gpu: int, dist_url: str) -> None:
    """Worker function for testing NCCL connectivity.

    Args:
        rank (int): Rank of the current process.
        num_gpu (int): Total number of GPUs.
        dist_url (str): Distributed URL for initialization.
    """
    import torch.distributed as dist

    dist.init_process_group(backend="NCCL", init_method=dist_url, rank=rank, world_size=num_gpu)
    dist.barrier(device_ids=[rank])


def main() -> None:
    """Main function to collect and print environment information."""
    global x

    print(collect_env_info())

    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        for k in range(num_gpu):
            device = f"cuda:{k}"
            try:
                x = torch.tensor([1, 2.0], dtype=torch.float32)
                x = x.to(device)
            except Exception as e:
                print(
                    f"Unable to copy tensor to device={device}: {e}. "
                    "Your CUDA environment is broken."
                )
        if num_gpu > 1:
            test_nccl_ops()


if __name__ == "__main__":
    main()
