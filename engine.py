# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import xywh2xyxy
import numpy as np


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, start_steps: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for batch in metric_logger.log_every(data_loader, print_freq, header):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // args.update_freq
        global_step = start_steps + step  # global training iteration
        img_data, text_data, target, obj_mask = batch
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)  # obj_mask shape:  torch.Size([96, 1, 224, 224])
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        # model forward
        pred_box, contrastive_loss, visu_sim, seg_mask, mlm_loss, mlm_acc, mlm_sts_pred, mim_pred, mim_vts_pred = \
            model(img_data.tensors, img_data.mask, text_data, global_step=global_step, training=True)
        loss_dict = loss_utils.one_ref_loss(args, pred_box, target, obj_mask, contrastive_loss, visu_sim, seg_mask,
                                            mlm_loss=mlm_loss)

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # The default value of max_norm is 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


"""
   The core training code is implemented here, which alternately models MIM and MLM.
   Implemented by Linhui Xiao.
     2024-01-10
"""
def train_one_epoch_with_mrefm(args, model: torch.nn.Module, vqkd: torch.nn.Module, data_loader: Iterable,
                               optimizer: torch.optim.Optimizer, device: torch.device,
                               epoch: int, start_steps: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 11  # ori: 10

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // args.update_freq
        global_step = start_steps + step  # global training iteration

        if global_step % 2 == 0:
            enable_ref_mim = True
            enable_ref_mlm = False
        else:
            enable_ref_mim = False
            enable_ref_mlm = True

        img_data, text_data, target, obj_mask, mim_img, mim_mask_pos, mim_vts_labels, mlm_sts_labels = batch
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)  # obj_mask shape:  torch.Size([96, 1, 224, 224])
        mim_img = mim_img.to(device)  # non_blocking=True)
        mim_mask_pos = mim_mask_pos.to(device)  # non_blocking=True)
        mim_vts_labels = mim_vts_labels.to(device)  # torch.Size([64, 576, 4])
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        if enable_ref_mim:
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                input_ids = vqkd.get_codebook_indices(mim_img)  # Tokenize the original image, torch.Size([24, 24, 24])
                bool_masked_pos = mim_mask_pos.flatten(1).to(torch.bool)  # numpy to torch, torch.Size([24, 576])
                mim_labels = input_ids[bool_masked_pos]  # Get the ID based on the mask, shape: torch.Size([5520]), 24*230=5520
        else:
            bool_masked_pos, mim_labels = None, None

        # model forward
        pred_box, contrastive_loss, visu_sim, seg_mask, mlm_loss, mlm_acc, mlm_sts_pred, mim_pred, mim_vts_pred = \
            model(img_data.tensors, img_data.mask, text_data, global_step=global_step, mim_masked_pos=bool_masked_pos,
                  obj_mask=obj_mask, enable_ref_mim=enable_ref_mim, enable_ref_mlm=enable_ref_mlm, training=True)
        # The `loss_dict` is a dictionary that contains `l1_smooth` and `giou`.
        loss_dict = loss_utils.one_ref_loss(args, pred_box, target, obj_mask, contrastive_loss, visu_sim, seg_mask,
                                            mim_pred, mim_labels, mim_vts_pred, mim_vts_labels,
                                            mlm_loss, mlm_sts_pred, mlm_sts_labels)

        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # The default value of max_norm is 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        pred_box, seg_mask, img_cls, text_cls = model(img_data.tensors, img_data.mask, text_data)
        miou, accu, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_val(args, pred_box, target, seg_mask, tgt_mask)

        metric_logger.update_v2('box_miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('box_accu', accu, batch_size)
        if mask_iou_list is not None:
            metric_logger.update_v2('seg_miou', torch.mean(mask_iou_list), batch_size)

        if args.use_mask_loss:
            metric_logger.update_v2('accu', torch.mean(mask_iou_list), batch_size)
        else:
            metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """ If the original text is passed in, uncomment out the following code. """
        # text_data = text_data.to(device)

        pred_box, seg_mask, img_cls, text_cls = model(img_data.tensors, img_data.mask, text_data)

        pred_box_list.append(pred_box.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    box_accuracy = float(result_tensor[0]) / float(result_tensor[1])

    if args.use_mask_loss:
        seg_miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        print("segmentation mIoU: ", seg_miou)
        seg_oiou = float(torch.sum(I_list, dim=0)) / float(torch.sum(U_list, dim=0))
        print("segmentation oIoU: ", seg_oiou)
        return seg_miou

    return box_accuracy


@torch.no_grad()
def evaluate_hivg(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    text_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        # text_data = text_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        """Core model calculation"""
        output, _, _, token_sim, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

        for text_i in text_data:
            text_list.append(text_i)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)

    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list, I_list, U_list = eval_utils.trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)


    """" Statistics the result with different text length """

    # statistic_diff_length_acc = True
    statistic_diff_length_acc = False
    # only can be used in one GPU，Using multiple cards will only print the result of a single card.
    if statistic_diff_length_acc:
        assert len(text_list) == iou.shape[0]
        count_for_len_in_1_to_5 = [0, 0]
        count_for_len_in_6_to_7 = [0, 0]
        count_for_len_in_8_to_10 = [0, 0]
        count_for_len_in_11_plus = [0, 0]
        for i in range(len(text_list)):
            len_i = len(text_list[i].split(" "))
            iou_i = iou[i]
            if (len_i >= 1) and (len_i <= 5):
                count_for_len_in_1_to_5[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_1_to_5[0] += 1
            elif (len_i >= 6) and (len_i <= 7):
                count_for_len_in_6_to_7[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_6_to_7[0] += 1
            elif (len_i >= 8) and (len_i <= 10):
                count_for_len_in_8_to_10[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_8_to_10[0] += 1
            elif (len_i >= 11):
                count_for_len_in_11_plus[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_11_plus[0] += 1

        print("acc in length  1-5: ", count_for_len_in_1_to_5, ", ",
              count_for_len_in_1_to_5[0] / count_for_len_in_1_to_5[1])
        print("acc in length  6-7: ", count_for_len_in_6_to_7, ", ",
              count_for_len_in_6_to_7[0] / count_for_len_in_6_to_7[1])
        print("acc in length 8-10: ", count_for_len_in_8_to_10, ", ",
              count_for_len_in_8_to_10[0] / count_for_len_in_8_to_10[1])
        print("acc in length  11+: ", count_for_len_in_11_plus, ", ",
              count_for_len_in_11_plus[0] / count_for_len_in_11_plus[1])

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print("accuracy2: ", accuracy)
    if args.use_mask_loss:
        miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        print("segmentation miou: ", miou)

    return accuracy


@torch.no_grad()
def evaluate_clip_vg(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, obj_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        output, _, _, _, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy

