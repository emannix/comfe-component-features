
import torch
import torch.nn.functional as F
import torch.nn as nn
from pdb import set_trace as pb
import numpy as np

from xformers.factory.model_factory import xFormerConfig
from ..xformer.factory import MyxFormer

from torch.nn import Conv2d

from torch.nn.init import trunc_normal_

from ...utils.visualize import visualize_mask, visualize_image, overlay_mask

class ComFeHead(nn.Module):

    def __init__(self, n_img_prototypes: int, 
            views = 2,
            num_classes = 21,
            num_classes_prototypes = 21,
            temp_pred=0.1,
            temp_pred2=0.1,
            temp_contrastive=0.1,
            label_smoothing = 0.1,
            # ==========================
            loss_pz=True,
            loss_pyz=True,
            loss_pyp=True,
            loss_patch_consistency=True,
            loss_image_contrast=True,
            loss_class_contrast=True,
            # ==========================
            patches=16,
            backbone_dim= 384,
            transformer_input_dim = 256,
            transformer_heads = 8,
            transformer_layers = 6,
            transformer_dropout = 0.1,
            transformer_attnpdropout = 0.0,
            # ==========================
            background_class = False,
            background_prototypes = 100,
            use_decoder_posenc = True,
            use_xformers = True,
            # ==========================
            one_hot_method = 'one_hot',
            **kwargs):
        super(ComFeHead, self).__init__()
        # =================================
        self.loss_pz = loss_pz
        self.loss_pyz = loss_pyz
        self.loss_pyp = loss_pyp
        self.loss_patch_consistency = loss_patch_consistency
        self.loss_image_contrast = loss_image_contrast
        self.loss_class_contrast = loss_class_contrast

        self.one_hot_method = one_hot_method
        # =================================
        self.n_img_prototypes = n_img_prototypes

        self.background_class = background_class
        self.background_prototypes = background_prototypes
        # =================================
        self.patches = patches

        self.use_decoder_posenc = use_decoder_posenc
        self.use_xformers = use_xformers

        self.backbone_dim = backbone_dim
        self.transformer_input_dim = transformer_input_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_dropout = transformer_dropout
        self.transformer_attnpdropout = transformer_attnpdropout
        # =================================
        self.views = views
        self.label_smoothing = label_smoothing
        # =================================
        self.temp_pred = temp_pred
        self.temp_pred2 = temp_pred2
        self.temp_contrastive = temp_contrastive
        # ==========================================
        # ==========================================
        if self.background_class:
            self.total_classes = num_classes + 1
            self.background_prototypes_use = self.background_prototypes
        else:
            self.total_classes = num_classes
            self.background_prototypes_use = 0
        self.num_classes = num_classes

        clusters_per_class = num_classes_prototypes//self.num_classes
        self.num_classes_prototypes = clusters_per_class*self.num_classes + self.background_prototypes_use
        self.clusters_per_class = clusters_per_class

        classes = torch.arange(self.total_classes).repeat(clusters_per_class)
        classes = torch.nn.functional.one_hot(classes, num_classes=self.total_classes)
        if self.background_class:
            background_matrix = torch.zeros((self.background_prototypes, self.total_classes))
            background_matrix[:,-1] = 1.0
            classes = torch.concat([classes, background_matrix])

        means = torch.randn( (classes.shape[0], self.backbone_dim))
        means = F.normalize(means, dim=1)
        self.register_parameter('cluster_class', torch.nn.Parameter(means))

        classes = classes* (1-label_smoothing) + label_smoothing/classes.shape[1]
        self.register_buffer('cluster_class_oh', classes)

        # ==========================================

        d_model = self.transformer_input_dim
        num_decoder_layers = self.transformer_layers
        # d_model = 256
        residual_norm_style = 'post'
        nhead = self.transformer_heads
        attention = 'scaled_dot_product'
        attn_pdrop = self.transformer_attnpdropout
        dropout = self.transformer_dropout
        activation = 'relu'
        hidden_layer_multiplier = self.transformer_heads
        pos_encoding_dim = self.patches**2  #32**2
        self.num_heads = nhead

        if self.use_xformers:
            decoder_config = [
                {
                    "reversible": False,  # Optionally make these layers reversible, to save memory
                    "block_type": "decoder",
                    "num_layers": num_decoder_layers,  # Optional, this means that this config will repeat N times
                    "dim_model": d_model,
                    'normalization': 'layernorm',
                    "residual_norm_style": residual_norm_style,  # Optional, pre/post
                    "multi_head_config_masked": {
                        "num_heads": nhead,
                        "residual_dropout": dropout,
                        "attention": {
                            "name": attention,
                            "dropout": attn_pdrop,
                            "causal": False,
                        },
                    },
                    "multi_head_config_cross": {
                        "num_heads": nhead,
                        "residual_dropout": dropout,
                        "attention": {
                            "name": attention,
                            "dropout": attn_pdrop,
                            "causal": False,
                        },
                    },
                    "feedforward_config": {
                        "name": "MLP",
                        "dropout": dropout,
                        "activation": activation,
                        "hidden_layer_multiplier": hidden_layer_multiplier,
                    },
                    # "position_encoding_config": {
                    #     "name": "sine",
                    #     "dim_model": pos_encoding_dim
                    # },
                    "position_encoding_config": {
                        "name": "learnable",
                        "seq_len": pos_encoding_dim,
                        "dim_model": d_model
                    }
                }
            ]

            # The ViT trunk
            config = xFormerConfig(decoder_config)
            self.transformer_decoder = MyxFormer.from_config(config)

            self.decoder_positional_encoding = self.transformer_decoder.decoders[0].pose_encoding.pos_emb
            self.transformer_decoder.decoders[0].pose_encoding = None
            self.transformer_norm = self.transformer_decoder.decoders[0].wrap_ff.norm
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
            self.register_parameter('decoder_positional_encoding', torch.nn.Parameter(torch.randn(1, pos_encoding_dim, d_model)* 0.02))
            self.transformer_norm = nn.LayerNorm(d_model)
            self.transformer_decoder.decoders = self.transformer_decoder.layers

        # =======================================

        # =======================================
        in_channels=self.backbone_dim
        feat_channels = self.transformer_input_dim
        out_channels = self.backbone_dim

        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        # =======================================

        num_queries = self.n_img_prototypes
        self.query_feat = nn.Embedding(num_queries, feat_channels)
        self.query_embed = nn.Embedding(num_queries, feat_channels)

        self.init_weights()

    # ============================================================

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        trunc_normal_(self.query_embed.weight, std=0.02)

    # ============================================================

    def predict_head(self, decoder_out, mask_feature, i):
        decoder_out = self.transformer_norm(decoder_out)

        mask_embed = self.mask_embed(decoder_out)

        mask_feature_use = F.normalize(mask_feature, dim=1)

        mask_embed = F.normalize(mask_embed, dim=2)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed/self.temp_pred, mask_feature_use)
        attn_mask = None

        return mask_pred, mask_embed, attn_mask

    def predict(self, x):
        x = x.reshape(-1, self.patches, self.patches, self.backbone_dim)

        x = x.permute(0,3,1,2)
        batch_size, _, h, w = x.shape

        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            mask_features = x
            num_transformer_feat_level = self.backbone_dim//self.transformer_input_dim
            multi_scale_memorys = torch.split(x, self.transformer_input_dim, dim=1)

            decoder_inputs = []
            decoder_positional_encodings = []
            for i in range(num_transformer_feat_level):
                decoder_input = multi_scale_memorys[i]
                # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
                decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
                decoder_positional_encoding = self.decoder_positional_encoding

                decoder_inputs.append(decoder_input)
                decoder_positional_encodings.append(decoder_positional_encoding)

            # shape (num_queries, c) -> (num_queries, batch_size, c)
            query_feat = self.query_feat.weight.unsqueeze(0).repeat(
                (batch_size, 1, 1))
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(
                (batch_size, 1, 1))

            query_pred_list = []
            mask_pred_list = []
            mask_pred, mask_embed, attn_mask = self.predict_head(query_feat, mask_features, 0)

            query_pred_list.append(mask_embed)
            mask_pred_list.append(mask_pred)
            for i in range(self.transformer_layers):
                level_idx = i % num_transformer_feat_level

                # cross_attn + self_attn
                layer = self.transformer_decoder.decoders[i]

                if self.use_decoder_posenc:
                    memory = decoder_inputs[level_idx] + decoder_positional_encodings[level_idx]
                    query = query_feat + query_embed
                else:
                    memory = decoder_inputs[level_idx]
                    query = query_feat

                if self.use_xformers:
                    query_feat = layer(target=query, memory= memory, encoder_att_mask= None)
                else:
                    query_feat = layer(query, memory)

                mask_pred, mask_embed, attn_mask = self.predict_head(
                    query_feat, mask_features, i+1)

                query_pred_list.append(mask_embed)
                mask_pred_list.append(mask_pred)

        return query_pred_list, mask_pred_list, mask_features

    # ============================================================

    def one_hot(self,y):
        if self.one_hot_method == 'one_hot':
            y_oh = torch.nn.functional.one_hot(y, num_classes = self.total_classes)
            if self.background_class:
                y_oh[:,-1] = 1.0
            return y_oh

    # ============================================================
    def forward_pred(self, x):
        query_pred_list, mask_pred_list, mask_features = self.predict(x)
        mask_logits = mask_pred_list[-1]
        hs_prototypes = query_pred_list[-1]

        mask_logits = mask_logits.permute(0,2,3,1)
        mask_logits = mask_logits.reshape(mask_logits.shape[0], -1, mask_logits.shape[-1])
        mask_probs_proto = mask_logits.softmax(dim=-1)

        cluster_class = F.normalize(self.cluster_class, dim=1)
        prototype_class_proto = hs_prototypes @ cluster_class.T /self.temp_pred2
        prototype_class_proto = prototype_class_proto.softmax(dim=-1)
        
        classes_oh = self.cluster_class_oh
        prototype_class = prototype_class_proto @ classes_oh

        mask_probs = torch.bmm(mask_probs_proto, prototype_class)
        mask_probs = mask_probs.reshape(-1, self.patches**2, mask_probs.shape[-1])
        mask_probs_global = mask_probs.max(dim=1)[0]

        mask_probs = mask_probs.reshape(-1, self.patches, self.patches, mask_probs.shape[-1])

        return mask_probs, mask_probs_global
    # ============================================================
    # ============================================================

    def forward(self, x, y, stage='fit', y_keep=None):

        query_pred_list, mask_pred_list, mask_features = self.predict(x)

        loss_pz = torch.tensor(0.0).to(x)
        loss_pyz = torch.tensor(0.0).to(x)
        loss_pyp = torch.tensor(0.0).to(x)
        loss_patch_consistency = torch.tensor(0.0).to(x)
        loss_image_contrast = torch.tensor(0.0).to(x)
        loss_class_contrast = torch.tensor(0.0).to(x)
        for i in range(len(mask_pred_list)):
            mask_logits = mask_pred_list[i]
            hs_prototypes = query_pred_list[i]

            mask_logits = mask_logits.permute(0,2,3,1)
            mask_logits = mask_logits.reshape(mask_logits.shape[0], -1, mask_logits.shape[-1])

            # per-image dimensions are NCHW
            mask_probs_proto = mask_logits.softmax(dim=-1)

            # calculate prototype class probabilities
            cluster_class = F.normalize(self.cluster_class, dim=1)
            prototype_class_proto_logits = hs_prototypes @ cluster_class.T /self.temp_pred2
            prototype_class_proto = prototype_class_proto_logits.softmax(dim=-1)

            classes_oh = self.cluster_class_oh

            prototype_class = prototype_class_proto @ classes_oh
            # =====================================================

            mask_probs = torch.bmm(mask_probs_proto, prototype_class)
            mask_probs = mask_probs.reshape(-1, self.patches**2, mask_probs.shape[-1])
            # =========================================
            mask_probs_global = mask_probs.max(dim=1)[0]
            if self.loss_pyz:
                if y is not None:
                    y_oh = self.one_hot(y)
                    denom = torch.clamp(1-mask_probs_global, min=1e-6, max=1.0)
                    mask_probs_global_logit = torch.log(mask_probs_global/denom)
                    loss_pyz += torch.nn.functional.binary_cross_entropy_with_logits(mask_probs_global_logit, y_oh.float(), reduction='none').sum(dim=1).mean()
            # ====================================================================
            if self.loss_pyp:
                if y is not None:
                    prototype_prob_global = prototype_class.max(dim=1)[0]
                    denom = torch.clamp(1-prototype_prob_global, min=1e-6, max=1.0)
                    prototype_prob_global_logit = torch.log(prototype_prob_global/denom)

                    y_oh = self.one_hot(y)
                    loss_pyp += torch.nn.functional.binary_cross_entropy_with_logits(prototype_prob_global_logit, y_oh.float(), reduction='none').sum(dim=1).mean()
            if self.loss_pyz is False:
                mask_probs_global = prototype_prob_global
            # ====================================================================
            if self.loss_image_contrast:
                if stage == 'fit':
                    z12 = hs_prototypes
                    hs_prototypes_use = torch.bmm(z12, z12.permute(0,2,1)) / self.temp_contrastive
                    hs_prototypes_use = hs_prototypes_use.reshape(-1, hs_prototypes_use.shape[-1])
                    hs_prototypes_use = hs_prototypes_use.log_softmax(dim=-1)

                    clusters_n = torch.arange(self.n_img_prototypes).to(x).long()
                    clusters = clusters_n.repeat(z12.shape[0])

                    loss_image_contrast += nn.NLLLoss()(hs_prototypes_use, clusters)
            # ====================================================================
            if self.loss_patch_consistency:
                if stage == 'fit':
                    # CARL loss
                    mask_probs_proto_use = mask_probs_proto.reshape(-1, mask_probs_proto.shape[-1])
                    z1, z2 = mask_probs_proto_use.split(mask_probs_proto_use.shape[0]//2)

                    loss_patch_consistency += -torch.mean(torch.log((z1*z2).sum(dim=1)))
            # ====================================================================
            if self.loss_class_contrast:
                contrast_probs = cluster_class @ cluster_class.T / self.temp_contrastive

                contrast_probs = contrast_probs.softmax(dim=1)
                contrast_probs = torch.clamp(contrast_probs, min=1e-6)

                cluster_y  = torch.nn.functional.one_hot(torch.arange(cluster_class.shape[0])).to(x)
                loss_class_contrast += torch.mean(torch.sum(-cluster_y*torch.log(contrast_probs), dim=1))
            # ====================================================================
            if self.loss_pz:
                loss_pz +=  - torch.mean(torch.logsumexp(mask_logits, dim=-1))

        denom = len(mask_pred_list)
        loss_pz = loss_pz/denom
        loss_pyz = loss_pyz/denom
        loss_pyp = loss_pyp/denom
        loss_patch_consistency = loss_patch_consistency/denom
        loss_image_contrast = loss_image_contrast/denom
        loss_class_contrast = loss_class_contrast/denom

        query_pred_list = query_pred_list[-1]
        mask_pred_list = mask_pred_list[-1]

        return (mask_probs_global, mask_probs, mask_probs_proto, query_pred_list, mask_features, prototype_class, prototype_class_proto), \
                (loss_pyz, loss_image_contrast, loss_patch_consistency, loss_pz, loss_class_contrast, loss_pyp)

    # =================================================
