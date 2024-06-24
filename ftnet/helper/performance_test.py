# https://deci.ai/the-correct-way-to-measure-inference-time-of-deep-neural-networks/"""
"""ImageTester Class for evaluating segmentation model inference time.

Example: python -m ftnet.helper.performance_test -c <path to .toml fie>
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..cfg import FTNetArgs, parse_args
from ..models import get_segmentation_model
from .logger import setup_logger

logger = logging.getLogger(__name__)
BatchNorm2d = nn.BatchNorm2d


class ImageTester:
    """Class to test segmentation model inference time.

    Args:
        args (argparse.Namespace): Hyperparameters for configuring the model and testing.

    Attributes:
        args (argparse.Namespace): Stored hyperparameters.
        device (torch.device): Device for model training and inference.
        logger (logging.Logger): Logger instance for logging messages.
        model (torch.nn.Module): Segmentation model instance.
        dummy_input (torch.Tensor): Dummy input tensor for inference time measurement.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")

        logger.info("Setting up the model")
        self.model = get_segmentation_model(
            model_name=self.args.model.name,
            dataset=self.args.dataset.name,
            backbone=self.args.model.backbone,
            norm_layer=nn.BatchNorm2d,
            dilated=self.args.model.dilation,
            no_of_filters=self.args.model.no_of_filters,
            pretrained_base=self.args.model.pretrained_base,
            edge_extracts=self.args.model.edge_extracts,
            num_blocks=self.args.model.num_blocks,
        )

        self.model.to(self.device)
        self.dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float).to(self.device)
        self.repetitions = 300

    def test(self):
        """Measure the inference time of the segmentation model.

        Performs inference multiple times and computes the mean
        inference time.
        """
        torch.set_grad_enabled(False)

        logger.info("\nEvaluation on Images:")
        self.model.eval()

        # INIT LOGGERS
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        timings = np.zeros((self.repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = self.model(self.dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in tqdm(range(self.repetitions)):
                starter.record()
                _ = self.model(self.dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                logger.debug(f"Repetition {rep+1} Run Time: {curr_time}")

        mean_syn = np.sum(timings) / self.repetitions
        std_syn = np.std(timings)
        logger.info(f"Average Run Time: {mean_syn}, Standard Deviation {std_syn}")

    def prepare(self, *args):
        """Prepare input tensors for model inference.

        Args:
            *args: Input tensors to be prepared.

        Returns:
            List[torch.Tensor]: List of prepared input tensors.
        """

        def _prepare(tensor):
            return tensor.to(self.device)

        return [_prepare(a) for a in args]


if __name__ == "__main__":
    args = parse_args()
    args = FTNetArgs.from_config(args.config)
    setup_logger(
        save_dir=None,
        distributed_rank=0,
        debug=False,
    )
    test_model = ImageTester(args)
    test_model.test()
