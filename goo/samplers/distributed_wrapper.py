from torch.utils.data.distributed import DistributedSampler
from .distributed_wrapper_alt import MyDistributedSamplerAlt
import torch.distributed as dist


def get_config_class():
    # Assume 'ENV' is an environment variable that can be 'PRODUCTION' or 'DEVELOPMENT'
    
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler
    else:
        return MyDistributedSamplerAlt


class MyDistributedSampler(get_config_class()):
    def __init__(self, dataset, num_replicas=None, rank=None, 
            shuffle=True, seed=0, drop_last=False, **kwargs):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

