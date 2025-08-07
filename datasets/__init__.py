# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import TransVGDataset
# from .data_loader_with_mim import OneRef_Dataset_with_MIM
from .data_loader_with_mim import OneRef_Dataset_with_MIM
from .masking_generator import BEIT3_MaskingGenerator, MAE_MaskingGenerator, Dynamic_MIM_MaskGenerator
from .masking_generator import Visual_Target_Relation_Score_Generator


""""CLIP's transform"""
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    if image_set in ['train', 'train_pseudo']:
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        # RandomResize default with_long_side = True
        # TODO: 不管如何压缩，都需要使用到长边压缩
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            # T.RandomHorizontalFlip(),  # This augmentation has a certain impact on performance and needs to be turned off. (xiaolinhui)
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])

    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')


class DataAugmentationForMIM(object):
    def __init__(self, args, image_set, is_onestage=False):
        imsize = args.imsize
        scales = []
        if args.aug_scale:  # default as open
            for i in range(6):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]
        # scales:  [384, 352, 320, 288, 256, 224, 192], Scales are bound to be smaller than the original size.

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        self.common_transform = T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            # T.RandomHorizontalFlip(),  # This augmentation has a certain impact on performance and needs to be turned off. (xiaolinhui)
            # T.ToTensor(),
            # T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])

        self.patch_transform = T.Compose([
            T.ToTensor(),
            T.NormalizeAndPad_FOR_MIM(size=imsize, aug_translate=args.aug_translate)
        ])

        # self.visual_token_transform = T.Compose([
        #     T.ToTensor(),
        #     T.WithoutNormAndPad(size=imsize, aug_translate=args.aug_translate)
        # ])

        # This function is a mask generation function. args.num_mask_patches = 75, with a default of 75 masks.
        # args.window_size= (224/16, 224/16)=14*14, 75/196=38%
        """" This is the Beit 3 masking method. """
        # self.masked_position_generator = BEIT3_MaskingGenerator(  #
        """" This is the MAE masking method. """
        # self.masked_position_generator = MAE_MaskingGenerator(  #
        """" This is the our Dynamic masking method. """
        self.masked_position_generator = Dynamic_MIM_MaskGenerator(  #
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,  # None，
            min_num_patches=args.min_mask_patches_per_block,  # = 16
            mim_mask_ratio=args.mim_mask_ratio,
            dynamic_mask_ratio=args.dynamic_mask_ratio,
        )

        self.visual_target_relation_score_generator = Visual_Target_Relation_Score_Generator(args.window_size)

    def __call__(self, input_dict):
        common_transform = self.common_transform(input_dict)
        """" Visual tokens. `for_patches` is passed to the model, which is the original input.
         `for_visual_tokens` is passed to the tokenizer, with only the normalization step omitted. """
        for_patches, for_visual_tokens = self.patch_transform(common_transform.copy())
        # for_visual_tokens = self.visual_token_transform(common_transform.copy())
        # mim_mask_pos = self.masked_position_generator()  # used for MAE，Beit3
        mim_mask_pos = self.masked_position_generator(for_patches.copy())  # used for Dynamic_MIM_MaskGenerator
        mim_vts_labels = self.visual_target_relation_score_generator(for_patches.copy())

        return for_patches, for_visual_tokens, mim_mask_pos, mim_vts_labels


# args.data_root default='./ln_data/', args.split_root default='data', '--dataset', default='referit'
# split = test, testA, val, args.max_query_len = 20
def build_dataset(split, args):
    if args.enable_ref_mim and split in ['train', 'train_pseudo']:
        return OneRef_Dataset_with_MIM(args,
                                       data_root=args.data_root,
                                       split_root=args.split_root,
                                       dataset=args.dataset,
                                       split=split,
                                       transform=DataAugmentationForMIM(args, split),
                                       max_query_len=args.max_query_len,
                                       prompt_template=args.prompt)
    else:
        return TransVGDataset(args,
                              data_root=args.data_root,
                              split_root=args.split_root,
                              dataset=args.dataset,
                              split=split,
                              transform=make_transforms(args, split),
                              max_query_len=args.max_query_len,
                              prompt_template=args.prompt)


