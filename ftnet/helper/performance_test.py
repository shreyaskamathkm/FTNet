# https://deci.ai/the-correct-way-to-measure-inference-time-of-deep-neural-networks/
import numpy as np
import torch
import torch.nn as nn
from core.models import get_segmentation_model
from core.utils import get_rank, setup_logger
from lightning_scripts.options import parse_args

BatchNorm2d = nn.BatchNorm2d


class ImageTester:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = torch.device("cuda")
        self.logger = setup_logger(
            name="pytorch_lightning",
            save_dir=None,
            distributed_rank=get_rank(),
            color=True,
            abbrev_name=None,
        )

        self.model = get_segmentation_model(
            model_name=self.hparams.model,
            dataset=self.hparams.dataset,
            backbone=self.hparams.backbone,
            norm_layer=BatchNorm2d,
            no_of_filters=self.hparams.no_of_filters,
            pretrained_base=self.hparams.pretrained_base,
            edge_extracts=self.hparams.edge_extracts,
            num_blocks=self.hparams.num_blocks,
        )
        self.model.to(self.device)
        self.dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float).to(
            self.device
        )

    def test(self):
        torch.set_grad_enabled(False)

        self.logger.info("\nEvaluation on Images:")
        self.model.eval()

        # INIT LOGGERS
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = self.model(self.dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.model(self.dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        # std_syn = np.std(timings)
        print(mean_syn)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.hparams.precision == "half":
                tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]


if __name__ == "__main__":
    hparams = parse_args()
    test_model = ImageTester(hparams)
    test_model.test()
