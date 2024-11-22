
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as pb

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, output_dim=1000, seed=None):
        super().__init__()
        num_classes = output_dim
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)

# ==============================================

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, 
            outputs='all', no_inference=False, train_feature_model=False, **kwargs):
        super().__init__()
        self.feature_model = feature_model
        self.train_feature_model = train_feature_model
        if train_feature_model == False:
            self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.half)
        self.autocast_ctx = autocast_ctx
        self.outputs = outputs
        self.no_inference = no_inference

        if self.train_feature_model == False:
            for param in self.feature_model.parameters():
                param.requires_grad = False

        # self.feature_model.cuda()
        # torch.manual_seed(0)
        # sample = torch.rand((1, 3, 224, 224)).cuda()
        # self.feature_model(sample)

    def forward(self, images):
        if self.train_feature_model == False:
            self.feature_model.eval()
            with torch.inference_mode():
                with self.autocast_ctx():
                    features = self.feature_model.get_intermediate_layers(
                        images, self.n_last_blocks, return_class_token=True
                    )
        else:
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                        images, self.n_last_blocks, return_class_token=True
                    )

        if self.outputs == 'all':
            features = features
        elif self.outputs == 'cls_tkn':
            features = features[0][1]
        elif self.outputs == 'patch_tkn':
            features = features[0][0]
        elif self.outputs == 'patch_tkn_noimg':
            features = features[0][0]
            features = features.reshape(-1, features.shape[2])
        elif self.outputs == 'all_combined':
            features = torch.concatenate([features[0][1].unsqueeze(1), features[0][0]], dim=1)
            
        if self.no_inference:
            if features.is_inference():
                features = torch.tensor(features).to(images)
        return features

    def forward_patch_embed(self, images):
        if self.train_feature_model == False:
            self.feature_model.eval()
            with torch.inference_mode():
                with self.autocast_ctx():
                    features = self.feature_model.patch_embed(images)
        else:
            with self.autocast_ctx():
                features = self.feature_model.patch_embed(images)

        if self.no_inference:
            if features.is_inference():
                features = torch.tensor(features).to(images)
        return features
