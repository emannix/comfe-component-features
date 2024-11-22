import torch

class DummySampler(torch.utils.data.Sampler):

    def __init__(
        self
    ):
    	self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

