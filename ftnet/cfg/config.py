# import argparse

import argparse

# def str2bool(v):
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False

#     raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FTNet Semantic Segmentation Training With Pytorch"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the TOML configuration file",
    )

    return parser.parse_args()
