# https://github.com/facebookresearch/suncet/blob/main/src/utils.py
import os
import math
import torch
import torch.distributed as dist

from pdb import set_trace as pb


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.,
        **kwargs
    ):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
            return new_lr / self.ref_lr

        if step <= (self.warmup_steps + self.T_max):
            # -- progress after warmup
            progress = float(step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        if step > (self.warmup_steps + self.T_max):
            new_lr = self.final_lr

        return new_lr / self.ref_lr