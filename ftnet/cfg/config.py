import argparse


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
