from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['Poly']


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, steps_per_epoch=0, warmup_epochs=0, power=0.9, last_epoch=-1):
        self.steps_per_epoch = steps_per_epoch
        self.cur_iter = 0
        self.power = power
        self.N = num_epochs * steps_per_epoch
        self.warmup_iters = warmup_epochs * steps_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.steps_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), self.power)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.steps_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]
