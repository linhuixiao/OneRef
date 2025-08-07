"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2020 Ross Wightman

Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np
import torch
from utils.box_utils import xywh2xyxy


"""
   Implemented by Linhui Xiao.
      2024-01-10
"""


class BEIT3_MaskingGenerator:
    def __init__(self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None, mask_ratio=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))  # log(0.3), log(10/3)
        # print("\nlog_aspect ratio: ", self.log_aspect_ratio)  # (-1.2039728043259361, 1.2039728043259361)

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        
        # maintain a fix number {self.num_masking_patches}
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_mask], mask_y[to_mask]] = 1

        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"

        return mask


class MAE_MaskingGenerator:
    def __init__(self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None, mask_ratio=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))  # log(0.3), log(10/3)

        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_simple(self):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mask_ratio
        N, L, D = 10, self.num_patches, 3  # batch, length, dim,  10 * N * 3
        x = torch.rand(N, L, D)
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def __call__(self):
        x_masked, mask, ids_restore = self.random_masking_simple()
        pick_number = random.randint(0, 9)
        mim_mask = mask[pick_number].reshape(self.height, self.width).numpy().astype(int)
        # print("mask =\n", mim_mask)  # 需要掩码的为 1

        return mim_mask


class Dynamic_MIM_MaskGenerator:
    def __init__(self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None, mim_mask_ratio=None, dynamic_mask_ratio=None, margin=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))  # log(0.3), log(10/3)

        self.mim_mask_ratio = mim_mask_ratio
        self.dynamic_mask_ratio = dynamic_mask_ratio
        self.margin = 1  # 2

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = self.mim_mask_ratio
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_simple(self, mrm_L, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # mask_ratio = self.mask_ratio
        N, L, D = 10, mrm_L, 3  # batch, length, dim,  10 * N * 3
        x = torch.rand(N, L, D)
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def __call__(self, input_dict):
        h, w = self.height, self.width
        box_norm = input_dict['box']  # tensor
        box_xywh = box_norm * torch.tensor([w, h, w, h], dtype=torch.float32)
        box_xyxy = xywh2xyxy(box_xywh).int().numpy()
        box_xyxy[0] = max(box_xyxy[0] - self.margin, 0)  # if box_xyxy[0] - self.margin >= 0 else box_xyxy[0]
        box_xyxy[1] = max(box_xyxy[1] - self.margin, 0)  # if box_xyxy[1] - self.margin >= 0 else box_xyxy[1]
        box_xyxy[2] = min(box_xyxy[2] + self.margin, w)  # if box_xyxy[2] + self.margin <= w else box_xyxy[2]
        box_xyxy[3] = min(box_xyxy[3] + self.margin, h)  # if box_xyxy[3] + self.margin <= h else box_xyxy[3]

        mrm_h, mrm_w = math.ceil(box_xyxy[3] - box_xyxy[1]), math.ceil(box_xyxy[2] - box_xyxy[0])
        mrm_L = mrm_h * mrm_w
        x_masked, mask, ids_restore = self.random_masking_simple(mrm_L, self.dynamic_mask_ratio)
        pick_number = random.randint(0, 9)
        mrm_mask = mask[pick_number].reshape(mrm_h, mrm_w).int()

        mim_x_masked, mim_mask, mim_ids_restore = self.random_masking_simple(h * w, self.mim_mask_ratio)
        pick_number = random.randint(0, 9)
        mim_mask = mim_mask[pick_number].reshape(h, w).int()
        mim_mask[box_xyxy[1]:box_xyxy[1] + mrm_h, box_xyxy[0]:box_xyxy[0] + mrm_w] = mrm_mask

        return mim_mask.numpy()


class Visual_Target_Relation_Score_Generator:
    def __init__(self, patch_length):
        if not isinstance(patch_length, tuple):
            patch_length = (patch_length,) * 2
        self.patch_height, self.patch_width = patch_length
        self.num_patches = self.patch_height * self.patch_width

    def __call__(self, input_dict):
        h, w = self.patch_height, self.patch_width
        box_norm = input_dict['box']  # tensor
        box_xcycwh = box_norm * torch.tensor([w, h, w, h], dtype=torch.float32)
        box_xcycwh = box_xcycwh.numpy()
        box_xcycwh[2] = max(box_xcycwh[2], 1)
        box_xcycwh[3] = max(box_xcycwh[3], 1)
        vts_score = torch.rand(self.patch_height, self.patch_width, 4)  # torch.Size([24, 24, 4])
        for j in range(self.patch_height):
            for i in range(self.patch_width):
                vts_score[j, i, :] = torch.tensor([(i - box_xcycwh[0]) / w, (j - box_xcycwh[1]) / h,
                                                  1 / box_xcycwh[2], 1 / box_xcycwh[3]],
                                                  dtype=torch.float32, device=vts_score.device)

        return vts_score.reshape(h * w, 4).numpy()


if __name__ == '__main__':
    import pdb
    generator = BEIT3_MaskingGenerator(input_size=14, num_masking_patches=118, min_num_patches=16,)

    mask = generator()

    for i in range(10000000):
        mask = generator()
        if mask.sum() != 118:
            pdb.set_trace()
            print(mask)
            print(mask.sum())
