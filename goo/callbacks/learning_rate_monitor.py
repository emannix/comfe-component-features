from lightning.pytorch.callbacks import LearningRateMonitor

from pdb import set_trace as pb


from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Type
from torch.optim.optimizer import Optimizer

class CustomLearningRateMonitor(LearningRateMonitor):

    def _get_lr_momentum_stat(self, optimizer: Optimizer, names: List[str]) -> Dict[str, float]:
        lr_momentum_stat = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults

        for pg, name in zip(param_groups, names):
            lr = self._extract_lr(pg, name)
            lr_momentum_stat.update(lr)
            momentum = self._extract_momentum(
                param_group=pg, name=name.replace(name, f"{name}-momentum"), use_betas=use_betas
            )
            lr_momentum_stat.update(momentum)
            if 'weight_decay' in pg:
                param = self._extract_custom(pg, 'weight_decay', name)
                lr_momentum_stat.update(param)
            if 'teacher_momentum' in pg:
                param = self._extract_custom(pg, 'teacher_momentum', name)
                lr_momentum_stat.update(param)

        return lr_momentum_stat


    def _extract_custom(self, param_group: Dict[str, Any], key: str, name: str) -> Dict[str, Any]:
        param = param_group[key]
        return {name + '-' + key: param}