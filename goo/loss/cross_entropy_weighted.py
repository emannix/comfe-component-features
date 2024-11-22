""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
from torch import nn
import numpy as np

from lightly.loss.memory_bank import MemoryBankModule
from lightly.utils import dist
from pdb import set_trace as pb
from torch.utils.data import Subset

# https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

class CrossEntropyWeighted(nn.Module):

    def __init__(
        self, beta=0.9999, reduction='none', **kwargs
    ):
        super(CrossEntropyWeighted, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.register_buffer('class_weights', torch.tensor(0.0), persistent=False)

    def initialise_weights(self, dataset, classes):
        if isinstance(dataset, Subset):
            targets = dataset.dataset.targets[dataset.indices]
        else:
            targets = dataset.targets

        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        class_counts = []
        for t in range(classes):
            class_counts.append(torch.sum(targets == t).item())

        class_counts = np.array(class_counts)
        class_counts[class_counts == 0] = 1 # if zero examples, set to 1

        # ==========================================
        beta = 0.9999

        class_weights = (1-self.beta)/(1-self.beta**class_counts)
        class_weights = class_weights/np.sum(class_weights)*classes
        self.class_weights = torch.tensor(class_weights).to(self.class_weights)

    def forward(self, y_hat, target):
        loss = nn.functional.cross_entropy(y_hat, target, weight=self.class_weights, reduction=self.reduction)
        return loss
