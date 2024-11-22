
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb
from typing import Any
import torch
import torch.nn.functional as F
import pandas as pd
import re
import numpy as np
import math
import copy
import torch.distributed as dist

from torchmetrics.clustering import RandScore, AdjustedMutualInfoScore, NormalizedMutualInfoScore, HomogeneityScore
from torchmetrics import JaccardIndex

from ...utils.visualize import visualize_mask, visualize_image, overlay_mask, overlay_mask_fixed
from PIL import Image
import cv2
from pathlib import Path
import os
import torchvision

from ...lightly.dataset import LightlyDataset
import torch.nn as nn
from tqdm import tqdm
import faiss

class ComFe(ModelBaseS):
    def __init__(self,
            predict_start_write = False,
            predict_start_exit_after = False,
            predict_start_dataset = 'train',
            plot_closest_training_data = False,
            plot_class_prototypes = False,
            plot_class_prototypes_n_exemplars = 1,
            plot_class_prototype_associations = False,
            **kwargs):
        super(ComFe, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.metrics = ['index', 'label', 'y_hat', 'loss_pz', 'loss_pyz', 'loss_pyp', 'loss_patch_consistency', 'loss_image_contrast', 'loss_class_contrast']
        self.metrics_log = [False, False, False, True,       True,      True,      True,      True,      True]
        self.metrics_save = [True, True, True, False,     False,     False,     False,     False,     False]
        
        self.backbone = self.networks.backbone()
        self.head = self.networks.base_clustering()

        self.num_classes = self.head.num_classes

        if self.head.background_class:
            self.total_classes = self.num_classes + 1
        else:
            self.total_classes = self.num_classes


    def model_step(self, batch, stage='fit'):

        imgs, y, idx = batch
        imgs = torch.cat(imgs)
        y = y.repeat(imgs.shape[0]//y.shape[0])

        embed, embed_global, embed_shape = self.forward(imgs)

        # ==================================
        (mask_probs_global, mask_probs, mask_probs_proto, query_pred_list, mask_features, prototype_class, prototype_class_proto), \
                (loss_pyz, loss_image_contrast, loss_patch_consistency, loss_pz, loss_class_contrast, loss_pyp)\
             = self.head(embed, y, stage)
        total_loss = loss_pyz + loss_image_contrast + loss_patch_consistency + loss_pz  + loss_class_contrast + loss_pyp
        # =========================

        if self.head.background_class:
            mask_probs_global = mask_probs_global[:, :-1]
        # =========================================
        results_dict = {
            'index': idx,
            'label': y,
            'y_hat': mask_probs_global,
            'loss': total_loss, 
            'loss_pyz': loss_pyz,
            'loss_image_contrast': loss_image_contrast,
            'loss_patch_consistency': loss_patch_consistency,
            'loss_pz': loss_pz,
            'loss_class_contrast': loss_class_contrast,
            'loss_pyp': loss_pyp
        }
        return results_dict

    # ==========================================================
    # ==========================================================

    def forward(self, x):
        embed = self.backbone(x)
        if isinstance(embed, tuple):
            embed_global = embed[0][1]
            embed = embed[0][0]
        else:
            embed_global = None

        embed_shape = embed.shape
        embed = embed.reshape(-1, embed.shape[-1])

        return embed, embed_global, embed_shape


# ============================================================================
# ============================================================================
# ============================================================================

    def on_predict_epoch_start( self ):
        if self.hparams.plot_closest_training_data or self.hparams.plot_class_prototypes or self.hparams.plot_class_prototype_associations:
            super(ComFe, self).on_predict_epoch_start()
            # iterate over batches and obtain prototypes
            datamodule = self.trainer.datamodule
            transform = datamodule.aug_predict

            dataloader_kNN = datamodule._predict_dataloader()
            if self.hparams.predict_start_dataset == 'train':
                dataset = copy.deepcopy(datamodule.data_train)
            elif self.hparams.predict_start_dataset == 'val':
                dataset = copy.deepcopy(datamodule.data_val)

            dataset = LightlyDataset.from_torch_dataset(dataset, transform, datamodule.aug_targets)
            self.ref_dataset = dataset
            dataloader_kNN.keywords['shuffle'] = False # obselete
            dataloader_kNN.keywords['drop_last'] = False # obselete
            dataloader_kNN = datamodule.predict_dataloader(dataset=dataset, base_dataloader=dataloader_kNN)

            train_data = dataloader_kNN

            if dist.is_initialized() and dist.get_world_size() > 1:
                rank = dist.get_rank()
            else:
                rank = 0
            miniters = self.trainer.log_every_n_steps
            if isinstance(self.trainer.limit_val_batches, int):
                self.limit_batches = True
            else:
                self.limit_batches = False

            self.dp = torch.tensor(0.0)
            self.dp = self.dp.to(next(self.backbone.parameters()).device)

            train_feature_bank = []
            train_class_proto_bank = []
            train_prototype_bank = []
            train_class_bank = []
            train_class_conf_bank = []
            train_img_idx = []

            with torch.no_grad():
                for i,data in enumerate(tqdm(train_data, position=rank, miniters=miniters)):
                    [img], target, idx = data
                    img, target = img.to(self.dp.device), target.to(self.dp.device)

                    proto_img, proto_label, proto_label_conf, present_classes, present_class_proto, proto_use, res = self.process_images(img, target, idx)

                    train_prototype_bank.append(present_classes)
                    train_class_proto_bank.append(present_class_proto)
                    train_feature_bank.append(proto_use.cpu())
                    train_class_bank.append(proto_label)
                    train_class_conf_bank.append(proto_label_conf)
                    train_img_idx.append(proto_img)

                    if self.limit_batches:
                        if i == self.trainer.limit_val_batches:
                            break

            self.train_feature_bank_all = self.process_data(train_feature_bank)
            self.train_prototype_bank_all = self.process_data(train_prototype_bank)
            self.train_class_proto_bank_all = self.process_data(train_class_proto_bank)
            self.train_class_bank_all = self.process_data(train_class_bank)
            self.train_class_conf_bank_all = self.process_data(train_class_conf_bank)
            self.train_img_idx_all = self.process_data(train_img_idx)


    def process_data(self, x):
        x = torch.cat(x, dim=0).contiguous()
        if dist.is_initialized() and dist.get_world_size() > 1:
            x_all = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(x_all, x)
            x_all = torch.cat(x_all, dim=0)
        else:
            x_all = x
        return x

    def process_images(self, x, y, idx):
        embed, embed_global, embed_shape = self.forward(x)
        (mask_probs_global, mask_probs, mask_probs_proto, query_pred_list, mask_features, prototype_class, prototype_class_proto), _ = self.head(embed, None, 'val')

        mask_probs_proto_present = mask_probs_proto.reshape(x.shape[0], self.head.patches**2, self.head.n_img_prototypes).argmax(dim=2)
        
        proto_class = torch.arange(self.head.n_img_prototypes).to(mask_probs_proto_present)
        cluster_class_oh = prototype_class.argmax(dim=2)

        present_classes = [mask_probs_proto_present[i].unique() for i in range(len(x))]
        present_class_proto = [prototype_class_proto[i][present_classes[i]] for i in range(len(x))]
        proto_label_conf = [prototype_class[i][present_classes[i]] for i in range(len(x))]
        proto_use = [query_pred_list[i][torch.isin(proto_class, present_classes[i])] for i in range(len(x))]
        proto_label = [cluster_class_oh[i][torch.isin(proto_class, present_classes[i])] for i in range(len(x))]
        proto_img = [ torch.tensor([idx[0][i]] * len(present_classes[i])) for i in range(len(x))]

        proto_use = torch.concat(proto_use, dim=0)
        proto_use = F.normalize(proto_use, dim=1)

        proto_label = torch.concat(proto_label, dim=0)#.argmax(dim=1)
        proto_label_conf = torch.concat(proto_label_conf, dim=0)#.argmax(dim=1)
        present_classes = torch.concat(present_classes, dim=0)
        proto_img = torch.concat(proto_img, dim=0).to(proto_label)

        present_class_proto = torch.concat(present_class_proto, dim=0)

        return proto_img, proto_label, proto_label_conf, present_classes, present_class_proto, proto_use, \
            (mask_probs_global, mask_probs, mask_probs_proto, query_pred_list, mask_features, prototype_class)

    def select_prototypes(self, embeddings, cluster_means, number=1):
        cluster_means = F.normalize(cluster_means)
        normed_embeddings = F.normalize(embeddings)
        normed_embeddings = normed_embeddings.clone().detach()
        cluster_means = cluster_means.clone().detach()

        index = faiss.IndexFlatIP(cluster_means.shape[1])
        index.add(np.ascontiguousarray(normed_embeddings.cpu()))
        D, I = index.search(cluster_means.cpu().numpy(), number)
        I = I.squeeze()
        return D, I

# =============================================================
    def predict_step(self, batch: Any, batch_idx: int):
        if self.hparams.predict_start_exit_after:
            exit()

        try:
            base_path = self.trainer.datamodule.data_val.root
            base_path_train = self.trainer.datamodule.data_train.root
        except:
            base_path = None
            base_path_train = None

        logger = self.trainer.logger
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'explanations')
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        total_prototypes = self.head.n_img_prototypes+1

        # ===============================================================================================
        # ===============================================================================================
        # visualizing class prototypes

        if self.hparams.plot_class_prototypes:
            class_prototypes = F.normalize(self.head.cluster_class, dim=1)
            class_prototypes_class = self.head.cluster_class_oh.argmax(dim=1).cpu()
            D,I = self.select_prototypes(self.train_feature_bank_all.to(class_prototypes), class_prototypes, number=self.hparams.plot_class_prototypes_n_exemplars)

            all_classes = class_prototypes_class.unique()
            # if self.base_clustering.background_class:
            #     all_classes = all_classes[:-1]

            D = D.squeeze()
            if len(D.shape) == 1:
                D = D[:, None]
            if len(I.shape) == 1:
                I = I[:, None]

            # find the all used class prototypes and plot these
            used_prototypes = self.train_class_proto_bank_all.argmax(dim=1).unique().cpu()
            used_prototypes_times = self.train_class_proto_bank_all.argmax(dim=1).unique(return_counts=True)[1].cpu().numpy()
            # used_prototypes = torch.arange(D.shape[0])
            all_class_prototypes = I[used_prototypes]
            all_classes_used = class_prototypes_class[used_prototypes]

            out_path = output_folder + '/prototypes/'
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            x = batch[0][0]
            if False:
                # for j in range(len(I)):
                #     ref_img, ref_y = self.vis_training_example(I, j, x, base_path_train, total_prototypes)
                #     ref_img.save(out_path + 'class_' + str(class_prototypes_class[j].item()) +'_refclass_' + str(ref_y) + '_prototype_' + str(j) +'.png')
                for j in range(len(all_class_prototypes)):
                    ref_img, ref_y = self.vis_training_example(all_class_prototypes[:,0], j, x, base_path_train, total_prototypes)
                    ref_img.save(out_path + 'class_' + str(all_classes_used[j].item()) +'_refclass_' + str(ref_y) + '_prototype_' + str(j) +'_frequency_'+str(used_prototypes_times[j])+'.png')

            # =====================================================================================================
            # find the prototypes belonging to each class and plot them together
            resize_size = 512
            crop_size = 512
            cc = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_size, max_size=resize_size+1),  # Resize the image
                torchvision.transforms.CenterCrop(crop_size)  # Center crop the resized image
            ])
            for i in range(len(all_classes)):
                ref_img_list = []
                counts = used_prototypes_times[all_classes_used == all_classes[i]]
                all_class_prototypes_i = all_class_prototypes[all_classes_used == all_classes[i]]
                all_class_prototypes_i = all_class_prototypes_i[(-counts).argsort()]
                if not isinstance(all_class_prototypes_i, np.ndarray):
                    all_class_prototypes_i = np.array([all_class_prototypes_i])
                if len(all_class_prototypes_i.shape) == 1:
                    all_class_prototypes_i = all_class_prototypes_i[:, None]

                # all_class_prototypes_i_use = all_class_prototypes_i[:,0]
                all_class_prototypes_i_use = all_class_prototypes_i.flatten()
                for j in range(len(all_class_prototypes_i_use)):
                    ref_img, ref_y = self.vis_training_example(all_class_prototypes_i_use, j, x, base_path_train, total_prototypes)
                    ref_img = torchvision.transforms.functional.pil_to_tensor(ref_img)
                    ref_img = cc(ref_img)
                    ref_img_list.append(ref_img)
                    # ref_img.save(out_path + 'class_' + str(class_prototypes_class.unique()[j].item()) +'_refclass_' + str(ref_y.item()) + '_prototype_' + str(j) +'.png')
                my_grid = torchvision.utils.make_grid(ref_img_list, nrow=D.shape[1])
                torchvision.utils.save_image(my_grid/255, out_path+'class_prototype_'+ str(all_classes[i].item())+'.png')
            # =====================================================================================================
            # find the closest prototype for each class and plot these
            resize_size = 128
            crop_size = 128
            cc = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_size, max_size=resize_size+1),  # Resize the image
                torchvision.transforms.CenterCrop(crop_size)  # Center crop the resized image
            ])
            best_class_prototypes = [I[:,0][class_prototypes_class == clss][D[:,0][class_prototypes_class == clss].argmax()] for clss in all_classes]
            ref_img_list = []
            for j in range(len(best_class_prototypes)):
                ref_img, ref_y = self.vis_training_example(best_class_prototypes, j, x, base_path_train, total_prototypes)
                ref_img = torchvision.transforms.functional.pil_to_tensor(ref_img)
                ref_img = cc(ref_img)
                ref_img_list.append(ref_img)
                # ref_img.save(out_path + 'class_' + str(class_prototypes_class.unique()[j].item()) +'_refclass_' + str(ref_y.item()) + '_prototype_' + str(j) +'.png')
            my_grid = torchvision.utils.make_grid(ref_img_list, nrow=20)
            torchvision.utils.save_image(my_grid/255, out_path+'class_prototype_closest_training_example.png')


        # ===============================================================================================
        # ===============================================================================================
        x = batch[0][0]
        y = batch[1]
        idx = batch[2][1]
        stage = 'eval'

        proto_img, proto_label, proto_label_conf, present_classes, present_class_proto, proto_use, res = self.process_images(x,y, batch[2])
        mask_probs_global, mask_probs, mask_probs_proto, query_pred_list, mask_features, prototype_class = res
        
        if self.head.background_class:
            correct = mask_probs_global[:, :-1].argmax(dim=1) == y
        else:
            correct = mask_probs_global.argmax(dim=1) == y
        # ============================================
        # # if stage == 'fit':
        # mask_probs_view = mask_probs.reshape(128, 16, 16, 11)
        mask_probs_view = mask_probs_proto.reshape(x.shape[0], self.head.patches, self.head.patches, self.head.n_img_prototypes)
        mask_probs_view = mask_probs_view.permute(0,3,1,2)

        probs_view = mask_probs.reshape(x.shape[0], self.head.patches, self.head.patches, self.total_classes)
        probs_view = probs_view.permute(0,3,1,2)

        for i in range(x.shape[0]):
            # =====================================================
            img = self.load_image_val(base_path, idx, i)

            query_pred_list_img = query_pred_list[i]
            prototype_class_img = prototype_class[i,:,:]
            probs_view_img = torch.nn.functional.interpolate(probs_view[i, None, :, :, :], size=(img.height, img.width), mode="bilinear")
            mask_probs_view_img = torch.nn.functional.interpolate(mask_probs_view[i, None, :, :, :], size=(img.height, img.width), mode="bilinear")
            
            mask_probs_view_img = mask_probs_view_img[0].argmax(dim=0)
            unique_vals, inverse_indices = torch.unique(mask_probs_view_img, return_inverse=True)
            # mask_probs_view_img = inverse_indices


            img_prototypes = overlay_mask_fixed(img, mask_probs_view_img, torch.arange(total_prototypes))
            img_class = overlay_mask_fixed(img, probs_view_img[0].argmax(dim=0), torch.arange(self.total_classes), background=self.head.background_class)


            out_path = output_folder + '/' + str(correct[i].cpu().numpy())  + '_' + str(y[i].cpu().numpy()) + '_' + idx[i].replace('/', '_')
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            img_prototypes.save(out_path + '_prototypes.png')
            img_class.save(out_path + '_class.png')
            img.save(out_path + '_raw.png')
            # img_class_proto.save(out_path + '_class_proto.png')


            # ===================================================================================
            # plotting heatmap image

            if self.head.background_class:
                mask_probs_global_img = mask_probs_global[i,:-1].argmax()
            else:
                mask_probs_global_img = mask_probs_global[i, :].argmax()
            # img_class_conf = visualize_image(probs_view_img[0, mask_probs_global_img])

            img_class_conf = probs_view_img[0, mask_probs_global_img]
            # img_class_conf[img_class_conf < 0.10] = 0
            # img_class_conf[mask_probs_global_img != probs_view_img[0].argmax(dim=0)] = 0

            heatmap_img = cv2.applyColorMap(255-(img_class_conf.cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2BGRA)
            target_color = (0, 0, 128, 255)
            mask = cv2.inRange(heatmap_img, target_color, target_color)
            heatmap_img[:,:,3] = 150
            # heatmap_img[mask > 0, 3] = 0

            pil_image = Image.fromarray(heatmap_img).convert("RGBA")

            combined = Image.alpha_composite(img.convert("RGBA"), pil_image)
            combined.save(out_path + '_class_conf.png')
            # ===================================================================================
            # ======================================================


            # ======================================================
            if self.hparams.plot_class_prototype_associations:
                img_idx = batch[2][0][i]
                img_class_proto_prob = present_class_proto[proto_img == img_idx]

                present_img_prototypes = mask_probs_view_img.unique()
                prototype_biggest = img_class_proto_prob.argmax(dim=1)
                prototype_biggest_score = img_class_proto_prob.max(dim=1)[0].cpu().numpy().round(decimals=5)
                I = prototype_biggest

                class_prototypes = F.normalize(self.head.cluster_class, dim=1)
                D,I = self.select_prototypes(self.train_feature_bank_all.to(class_prototypes), class_prototypes[I], number=1)

                I_class = self.train_class_bank_all[I].cpu().numpy()
                # I = [I]
                x = batch[0][0]
                for j in range(len(I)):
                    ref_img, ref_y = self.vis_training_example(I, j, x, base_path_train, total_prototypes)
                    comparison_correct = str((ref_y == y[i]).cpu().numpy())
                    ref_img.save(out_path + '_class_proto_match_' + comparison_correct +'_' + str(ref_y) +'_' + str(j) +'_class_'+str(I_class[j]) + '_score'+str(prototype_biggest_score[j]) +'.png')

            # ======================================================
            if self.hparams.plot_closest_training_data:
                img_idx = batch[2][0][i]
                img_proto_use = proto_use[proto_img == img_idx]
                img_present_classes = present_classes[proto_img == img_idx]
                img_proto_label = proto_label[proto_img == img_idx]

                D,I = self.select_prototypes(self.train_feature_bank_all.to(x), img_proto_use)

                if I.shape == ():
                    I = np.array([i])

                for j in range(len(I)):

                    ref_img, ref_y = self.vis_training_example(I, j, x, base_path_train, total_prototypes)
                    
                    comparison_correct = str((ref_y == y[i]).cpu().numpy())
                    ref_img.save(out_path + '_nearest_match_' + comparison_correct +'_' + str(ref_y) +'_' + str(j) +'.png')

        # ============================================

        return mask_probs_global


        # ============================================

    def vis_training_example(self, I, j, x, base_path_train, total_prototypes):
        ref_img, ref_y, ref_idx = self.ref_dataset.__getitem__(self.train_img_idx_all[I[j]]) 

        ref_embed, _, _ = self.forward(ref_img[0][None,:,:,:].to(x))
        (_, _, ref_mask_probs_proto, _, _, _, _), _ = self.head(ref_embed, None, 'val')
        ref_mask_probs_proto_view = ref_mask_probs_proto.reshape(1, self.head.patches, self.head.patches, self.head.n_img_prototypes)
        ref_mask_probs_proto_view = ref_mask_probs_proto_view.permute(0, 3, 1, 2)

        ref_img =  self.load_image_train(base_path_train, ref_idx)

        ref_mask_probs_proto_view = torch.nn.functional.interpolate(ref_mask_probs_proto_view, size=(ref_img.height, ref_img.width), mode="bilinear")
        ref_mask_probs_proto_view = ref_mask_probs_proto_view.argmax(dim=1) == self.train_prototype_bank_all[I[j]].item()
        ref_mask_probs_proto_view = ref_mask_probs_proto_view.float()*2 - 1
        
        ref_mask_probs_proto_view[ref_mask_probs_proto_view > 1e-6] = self.train_prototype_bank_all[I[j]].item()
        ref_mask_probs_proto_view[ref_mask_probs_proto_view < -1e-6] = self.head.n_img_prototypes
        ref_mask_probs_proto_view = ref_mask_probs_proto_view.int()
        ref_img = overlay_mask_fixed(ref_img, ref_mask_probs_proto_view[0], torch.arange(total_prototypes), binary=True)

        if isinstance(ref_y, torch.Tensor):
            ref_y = int(ref_y)
        return ref_img, ref_y


    def load_image_train(self, base_path_train, ref_idx):
        try:
            try:
                try:
                    try:
                        try:
                            try:
                                try:
                                    ref_img = Image.open(base_path_train + ref_idx[1])
                                except:
                                    ref_img = Image.open(self.trainer.datamodule.data_train.images[int(ref_idx[0])])
                            except:
                                ref_img = Image.open(self.trainer.datamodule.data_train._images[int(ref_idx[0])])
                        except:
                            ref_img = Image.open(self.trainer.datamodule.data_train._image_files[int(ref_idx[0])])
                    except:
                        ref_img = self.trainer.datamodule.data_train.data[ref_idx[0]]
                        ref_img = torch.tensor(ref_img).permute(2, 0, 1)
                        ref_img = torch.nn.functional.interpolate(ref_img[None,:,:,:], size=(224, 224), mode="bilinear")
                        ref_img = torchvision.transforms.functional.to_pil_image(ref_img[0])
                except:
                    ref_img = Image.open(self.trainer.datamodule.data_train._samples[int(ref_idx[0])][0])
            except:
                ref_img = Image.open(self.trainer.datamodule.data_train.inputs[int(ref_idx[0])])
        except:
            ref_img = Image.open(self.trainer.datamodule.data_train.dataset.inputs[int(ref_idx[0])])
        return ref_img

    def load_image_val(self, base_path, idx, i):
        try:
            try:
                try:
                    try:
                        try:
                            try:
                                try:
                                    img = Image.open(base_path + idx[i])
                                except:
                                    img = Image.open(self.trainer.datamodule.data_test.images[int(idx[i])])
                            except:
                                img = Image.open(self.trainer.datamodule.data_test._images[int(idx[i])])
                        except:
                            img = Image.open(self.trainer.datamodule.data_test._image_files[int(idx[i])])
                    except:
                        img = self.trainer.datamodule.data_test.data[int(idx[i])]
                        img = torch.tensor(img).permute(2, 0, 1)
                        img = torch.nn.functional.interpolate(img[None,:,:,:], size=(224, 224), mode="bilinear")
                        img = torchvision.transforms.functional.to_pil_image(img[0])
                except:
                    img = Image.open(self.trainer.datamodule.data_test._samples[int(idx[i])][0])
            except:
                img = Image.open(self.trainer.datamodule.data_test.inputs[int(idx[i])])
        except:
            img = Image.open(self.trainer.datamodule.data_test.dataset.inputs[int(idx[i])])
            
        
        return img