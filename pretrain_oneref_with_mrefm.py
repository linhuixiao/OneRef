import os
import time
import math
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
# from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, train_one_epoch_with_mrefm, validate

""" DO NOT delete the below OneRef model import code ! """
from timm.models import create_model
import models.utils as beit3_utils
import models.OneRef_model as OneRef_model
from models.utils import NativeScalerWithGradNormCount as NativeScaler
import models.modeling_vqkd as modeling_vqkd


def get_args_parser():
    parser = argparse.ArgumentParser('OneRef Args', add_help=False)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--lr_exponential', default=0.9, type=float, help='lr exponential')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true', help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true', help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true', help="If true, use random translate augmentation")
    # BEiT-3 Args
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, default='grounding',
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps',
                                 'imagenet', 'grounding'], help='Name of task to fine-tuning')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str,
                        default='/hdd/lhxiao/beit3/checkpoint/beit3.spm',
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--eval_batch_size', default=None, type=int)
    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--enable_seg_mask', action='store_true', help="If true, use segmentation mask, otherwise use box mask.")
    parser.add_argument('--frozen_backbone', action='store_true', default=False)
    parser.add_argument('--use_contrastive_loss', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_box_mask_constraints', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_mask_loss', action='store_true', help="If true, use segmentation loss")
    parser.add_argument('--use_regress_box', action='store_true', help="If true, enable regress box loss")
    # MRefM Args
    parser.add_argument('--enable_ref_mlm', action='store_true', help="If true, use mlm loss", default=False)
    parser.add_argument('--enable_ref_mim', action='store_true', help="If true, use mim loss", default=False)
    parser.add_argument('--enable_dynamic_mim', action='store_true', help="If true, use mim loss", default=False)
    parser.add_argument('--enable_mim_vts', action='store_true', help="If true, use mim visual target relation score", default=False)
    parser.add_argument('--enable_mlm_sts', action='store_true', help="If true, use mlm semantic target relation score", default=False)
    parser.add_argument('--enable_mrefm', action='store_true', help="If true, use milm loss", default=False)
    parser.add_argument('--text_mask_prob', type=float, default=0.4)  # import args, 0.5 in beit3，mlm_mask_ratio
    parser.add_argument('--mim_mask_ratio', type=float, default=0.35)  # import args
    parser.add_argument('--dynamic_mask_ratio', type=float, default=0.75)  # import args

    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--update_freq', default=1, type=int)

    # mim pretraining
    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--mim_mid_layer', default=0, type=int, help='mim_mid_layer,set 0 or 9')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='hidden dimension of codebook')
    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str, default="/hdd/lhxiao/beit2/checkpoint/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth")
    parser.add_argument("--tokenizer_model", type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")

    parser.add_argument('--num_mask_patches', default=75, type=int, help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data/image_data/', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')
    # Prompt Engineering: "{pseudo_query}" denote without using prompt
    #                    "{pseudo_query}" or using "find the region that corresponds to the description {pseudo_query}"
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size,
            code_dim=args.codebook_dim,
        ).eval()
    return model


def main(args):
    """ distribution init """
    if args.enable_mrefm:
        args.enable_ref_mim = True
        args.enable_ref_mlm = True

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('### INFO ### torch.backends.cudnn.benchmark = {}'.format(torch.backends.cudnn.benchmark))

    # If the suffix of the model does not include the task, then include the task name
    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval"):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        elif args.task in ("grounding"):
            model_config = "%s_grounding" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    print("model_config = %s" % model_config)

    """ Generate the OneRef model """
    model = create_model(model_config,
                         sys_args=args,
                         pretrained=False,
                         drop_path_rate=args.drop_path,
                         vocab_size=args.vocab_size,  # Vocabulary size: default 64010
                         checkpoint_activations=args.checkpoint_activations,)

    # Determine the window size when using the mask
    patch_size = model.beit3.vision_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.imsize // patch_size[0], args.imsize // patch_size[1])  # 384/16=24*24
    args.patch_size = patch_size  # 16*16
    args.num_mask_patches = int((args.window_size[0] * args.window_size[1]) * args.mim_mask_ratio)
    # prepare visual tokenizer
    vqkd = get_visual_tokenizer(args).to(device)

    print(args)

    if args.finetune:
        # Load the checkpoint. The model_key is the name of the key contained in the model dictionary package:
        # 'model|module'. The main purpose of this function is to perform upsampling interpolation for the
        # position embedding.
        beit3_utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param, "lr": args.lr},
                  {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                  {"params": visu_tra_param, "lr": args.lr_visu_tra},
                  {"params": text_tra_param, "lr": args.lr_bert}]
    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supported')

    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'exponential':
        lr_func = lambda epoch: args.lr_exponential ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError('Lr scheduler type not supported')

    # build dataset
    print('build dataset...')
    if (args.sup_type == 'full'):
        print("perform fullly supervised setting.")
        dataset_train = build_dataset('train', args)
    else:  # un
        print("perform unsupervised setting.")
        dataset_train = build_dataset('train_pseudo', args)

    # note certain dataset does not have 'test' set: eg. 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    dataset_val = build_dataset('val', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    """ Note: MRefM pretraining need use new dataloader utils.collate_fn_mim"""
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_mim, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("Tokenizer = %s" % str(vqkd))
    print("LR = %.8f" % args.lr)
    print("Total batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    best_accu = 0

    if args.finetune and not args.resume:
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("Init finetune accu: {}".format(best_accu))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys when loading resume model: \n', missing_keys)
        print('Unexpected additional keys in resume model: \n', unexpected_keys)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("Resume best_accu: {}".format(best_accu))

    if args.retrain:
        # --retrain used for testing "retrain the model",
        # according to paper: SiRi：A Simple Selective Retraining Mechanism for Transformer-based VG, ECCV 2022
        # However, results shows no gains for pretrained model.
        model_cache = build_model(args)
        model_cache.to(device)
        checkpoint = torch.load(args.retrain, map_location='cpu')
        model_cache.load_state_dict(checkpoint['model'])
        model_without_ddp.vl_transformer = model_cache.vl_transformer

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_ep_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_with_mrefm(args, model, vqkd, data_loader_train, optimizer, device, epoch,
                                                 epoch * num_training_steps_per_epoch, args.clip_max_norm)
        lr_scheduler.step()
        val_stats = validate(args, model, data_loader_val, device)
        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'n_parameters': n_parameters}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            if args.enable_ref_mlm or args.enable_ref_mim:
                checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
                if (epoch + 1) % 10 == 0:
                    checkpoint_paths.append(os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(str(epoch))))
                # checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(str(epoch)))]
            else:
                checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(os.path.join(args.output_dir, 'best_checkpoint.pth'))
                best_accu = val_stats['accu']

            for checkpoint_path in checkpoint_paths:
                print('Checkpoint is saving to: ', str(checkpoint_path))
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)

        end_ep_time = time.time()
        total_time = end_ep_time - start_ep_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Current epoch training time {}'.format(total_time_str))
        print('Checkpoints have been saved!')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OneRef training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
