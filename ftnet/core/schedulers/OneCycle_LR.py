import math

from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["OneCycle"]


class OneCycle(_LRScheduler):
    def __init__(
        self,
        optimizer,
        num_epochs,
        iters_per_epoch=0,
        last_epoch=-1,
        momentums=(0.85, 0.95),
        div_factor=25,
        phase1=0.3,
    ):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.phase1_iters = int(self.N * phase1)
        self.phase2_iters = self.N - self.phase1_iters
        self.momentums = momentums
        self.mom_diff = momentums[1] - momentums[0]
        self.low_lrs = [opt_grp["lr"] / div_factor for opt_grp in optimizer.param_groups]
        self.final_lrs = [opt_grp["lr"] / (div_factor * 1e4) for opt_grp in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1

        # Going from base_lr / 25 -> base_lr
        if self.phase1_iters >= T:
            cos_anneling = (1 + math.cos(math.pi * T / self.phase1_iters)) / 2
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]["momentum"] = (
                    self.momentums[0] + self.mom_diff * cos_anneling
                )

            return [
                base_lr - (base_lr - low_lr) * cos_anneling
                for base_lr, low_lr in zip(self.base_lrs, self.low_lrs)
            ]

        # Going from base_lr -> base_lr / (25e4)
        T -= self.phase1_iters
        cos_anneling = (1 + math.cos(math.pi * T / self.phase2_iters)) / 2

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]["momentum"] = (
                self.momentums[1] - self.mom_diff * cos_anneling
            )
        return [
            final_lr + (base_lr - final_lr) * cos_anneling
            for base_lr, final_lr in zip(self.base_lrs, self.final_lrs)
        ]
