
import torch
from pdb import set_trace as pb
import types

def weight_decay_exclude(name, param, param_wd_exclude, drop_1dim):
    if drop_1dim:
        return any(nd in name for nd in param_wd_exclude) or param.ndim <= 1
    else:
        return any(nd in name for nd in param_wd_exclude)

def LARS_exclude(name, param, param_lars_exclude, drop_1dim):
    if drop_1dim:
        return any(nd in name for nd in param_lars_exclude) or param.ndim <= 1
    else:
        return any(nd in name for nd in param_lars_exclude)

def set_parameter_groups(model, named_parameters, 
        param_wd_exclude=None, param_lars_exclude=None, param_opt_exclude=None,
        drop_1dim=False, **kwargs):
    if 'optimizer' in model.hparams.optimizer.keywords and model.hparams.optimizer.keywords['optimizer'] is not None:
        weight_decay = model.hparams.optimizer.keywords['optimizer'].keywords['weight_decay']
    else:
        weight_decay = model.hparams.optimizer.keywords['weight_decay']

    for network in model.children():
        if hasattr(network, 'no_weight_decay'):
            if param_wd_exclude is not None:
                param_wd_exclude = param_wd_exclude + list(network.no_weight_decay())
            else:
                param_wd_exclude = list(network.no_weight_decay())
                
    if isinstance(named_parameters, types.GeneratorType):
        named_parameters = [(n,p) for n, p in named_parameters]
    named_parameters = [(n,p) for n, p in named_parameters if p.requires_grad]

    if param_opt_exclude is not None:
        named_parameters = [(n,p) for n, p in named_parameters if not any(nd in n for nd in param_opt_exclude)]

    if param_wd_exclude is not None or param_lars_exclude is not None:
        grouped_parameters = [
            {'params': [p for n, p in named_parameters if (weight_decay_exclude(n,p,param_wd_exclude,drop_1dim) and LARS_exclude(n,p,param_lars_exclude, drop_1dim))], 
            'weight_decay': 0.0, 'layer_adaptation': False, 'LARS_exclude': True},
            {'params': [p for n, p in named_parameters if  (weight_decay_exclude(n,p,param_wd_exclude,drop_1dim) and not LARS_exclude(n,p,param_lars_exclude, drop_1dim))], 
            'weight_decay': 0.0, 'layer_adaptation': True, 'LARS_exclude': False},
            {'params': [p for n, p in named_parameters if  (not weight_decay_exclude(n,p,param_wd_exclude,drop_1dim) and LARS_exclude(n,p,param_lars_exclude, drop_1dim))], 
            'weight_decay': weight_decay, 'layer_adaptation': False, 'LARS_exclude': True},
            {'params': [p for n, p in named_parameters if (not weight_decay_exclude(n,p,param_wd_exclude,drop_1dim) and not LARS_exclude(n,p,param_lars_exclude, drop_1dim))], 
            'weight_decay': weight_decay, 'layer_adaptation': True, 'LARS_exclude': False},
        ]
    else:
        grouped_parameters = [{'params': [p for n, p in named_parameters]}]
        # grouped_parameters = [x for x in model.named_parameters()]
    # res1 = [n for n, p in model.opt_parameters()]
    # res2 = [n for n, p in model.named_parameters()]
    # pb()
    return grouped_parameters

