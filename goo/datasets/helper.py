import os
import omegaconf
import pandas as pd
import torch
import numpy as np
from pdb import set_trace as pb

from .iteround import saferound

def set_base_dir(base_dir):
    if os.environ.get('mysystem') is not None:
        system = os.environ.get('mysystem')
        if system == 'gadi':
            if os.environ.get('PBS_JOBFS') is not None:
                base_dir = os.environ.get('PBS_JOBFS') + '/'
        elif system == 'spartan':
            base_dir = '/tmp/'
    return base_dir

def combine_directories(base_path, target_path):
    if type(target_path) is omegaconf.listconfig.ListConfig:
        target_list = []
        for i in range(len(target_path)):
            target_list.append(base_path + target_path[i])
    else:
        target_list = base_path + target_path
    return target_list


class MySubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.dataset.targets = np.array(self.dataset.targets)
        self.indices = np.sort(indices)

    @property
    def transform(self):
        """Getter for the transform of the dataset."""
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        """Setter for the transform of the dataset."""
        self.dataset.transform = t

    @property
    def targets(self):
        """Getter for the transform of the dataset."""
        return self.dataset.targets[self.indices]

    @property
    def classes(self):
        """Getter for the transform of the dataset."""
        return np.unique(self.dataset.targets[self.indices])


def load_indices_from_csv(path):
    picked_indices_csv = pd.read_csv(path)
    picked_indices = picked_indices_csv['indices'].values
    return picked_indices

# ==============================================================

def random_indices(dataset, params, seed):
    if params['method'] == 'random_labelled_indices':
        return random_labelled_indices(dataset, params['number'], seed)
    elif params['method'] == 'random_nobalance_indices':
        return random_nobalance_indices(dataset, params['number'], seed)
    elif params['method'] == 'random_labelled_stratified_indices':
        return random_labelled_stratified_indices(dataset, params['number'], seed)
# ==============================================================

def random_nobalance_indices(dataset, number, seed):
    targets = np.array(dataset.targets)
    targets_index = np.arange(targets.shape[0])

    gen = np.random.default_rng(seed=seed)

    pot_idx = np.arange(targets_index.shape[0])
    if number > targets_index.shape[0]:
        idx = pot_idx
    else:
        idx = gen.choice(pot_idx, size=number, replace=False)
    print(idx)
    return idx

def random_labelled_indices(dataset, number, seed):
    targets = np.array(dataset.targets)

    targets_unique = np.unique(targets)
    targets_index = np.arange(targets.shape[0])

    gen = np.random.default_rng(seed=seed)
    number_per_class = number//targets_unique.shape[0]

    indices_list = []
    for i in range(len(targets_unique)):
        potential_idx = targets_index[targets == targets_unique[i]]
        # ======================================
        pot_idx = np.arange(potential_idx.shape[0])
        idx = gen.choice(pot_idx, size=number_per_class, replace=False)
        # ======================================
        potential_idx = potential_idx[idx]
        indices_list.append(potential_idx)
    idx_all = np.concatenate(indices_list)
    print(idx_all)
    print(targets[idx_all])
    return idx_all

def random_labelled_stratified_indices(dataset, number, seed):
    targets = np.array(dataset.targets)

    targets_unique = np.unique(targets, return_counts=True)
    targets_count = targets_unique[1]
    targets_prop = targets_count/np.sum(targets_count)
    targets_unique = targets_unique[0]
    targets_index = np.arange(targets.shape[0])

    gen = np.random.default_rng(seed=seed)
    # number_per_class = number//targets_unique.shape[0]
    # number_per_class = targets_prop*number
    number_per_class = np.array(saferound(targets_prop*number, 0)).astype(int)

    indices_list = []
    for i in range(len(targets_unique)):
        potential_idx = targets_index[targets == targets_unique[i]]
        # ======================================
        pot_idx = np.arange(potential_idx.shape[0])
        idx = gen.choice(pot_idx, size=number_per_class[i], replace=False)
        # ======================================
        potential_idx = potential_idx[idx]
        indices_list.append(potential_idx)
    idx_all = np.concatenate(indices_list)
    print(idx_all)
    print(targets[idx_all])
    return idx_all