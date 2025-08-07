import mpmath
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size, mdetr_interpolate


def build_target(args, gt_bbox, pred, device):
    batch_size = gt_bbox.size(0)
    num_scales = len(pred)
    coord_list, bbox_list = [], []
    for scale_ii in range(num_scales):
        this_stride = 32 // (2 ** scale_ii)
        grid = args.size // this_stride
        # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
        center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
        center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
        box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
        coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
        # Normalized by the image size
        coord = coord / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch_size):
        anch_ious = []
        for scale_ii in range(num_scales):
            this_stride = 32 // (2 ** scale_ii)
            grid = args.size // this_stride
            gw = coord_list[scale_ii][ii, 2]
            gh = coord_list[scale_ii][ii, 3]

            anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
            anchors = [args.anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [(x[0] / (args.anchor_imsize / grid),
                               x[1] / (args.anchor_imsize / grid)) for x in anchors]

            gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
            ## Get shape of anchor box
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))

            ## Calculate iou between gt and anchor shapes
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))
        best_scale = best_n // 3

        best_grid = args.size // (32 / (2 ** best_scale))
        anchor_idxs = [x + 3 * best_scale for x in [0, 1, 2]]
        anchors = [args.anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (args.anchor_imsize / best_grid), \
                           x[1] / (args.anchor_imsize / best_grid)) for x in anchors]

        gi = coord_list[best_scale][ii, 0].long()
        gj = coord_list[best_scale][ii, 1].long()
        tx = coord_list[best_scale][ii, 0] - gi.float()
        ty = coord_list[best_scale][ii, 1] - gj.float()
        gw = coord_list[best_scale][ii, 2]
        gh = coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled_anchors[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n % 3][1] + 1e-16)

        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    for ii in range(len(bbox_list)):
        bbox_list[ii] = bbox_list[ii].to(device)
    return bbox_list, best_gi, best_gj, best_n_list


def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1. / 5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    num_scale = len(pred_list)
    batch_size = pred_list[0].size(0)

    pred_bbox = torch.zeros(batch_size, 4).to(device)
    gt_bbox = torch.zeros(batch_size, 4).to(device)
    for ii in range(batch_size):
        pred_bbox[ii, 0:2] = torch.sigmoid(
            pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred_list[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[best_n_list[ii] // 3][ii, best_n_list[ii] % 3, :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    pred_conf_list, gt_conf_list = [], []
    for scale_ii in range(num_scale):
        pred_conf_list.append(pred_list[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
        gt_conf_list.append(target[scale_ii][:, :, 4, :, :].contiguous().view(batch_size, -1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf


# Contrastive Loss
class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss


# The below code is copied from transformers/models/clip/modeling_clip.py
# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def one_ref_loss(args, batch_pred, batch_target, tgt_mask, contrastive_loss, visu_sim=None, seg_mask=None,
                 mim_pred=None, mim_labels=None, mim_vts_pred=None, mim_vts_labels=None,
                 mlm_loss=None, mlm_sts_pred=None, mlm_sts_labels=None):
    """Compute the losses related to the bounding boxes,
       including the L1 regression loss and the GIoU loss
    """

    batch_size = batch_pred.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))

    losses = {}
    if args.use_regress_box:
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

    if args.use_contrastive_loss:
        # losses['loss_contrastive'] = (contrastive_loss / num_boxes) * 10.0
        losses['loss_contrastive'] = contrastive_loss / num_boxes
        """DO NOT multiply by 10, or the performance will drop sharply."""
        # losses['loss_contrastive'] = contrastive_loss * 100.0
        # losses['loss_contrastive'] = contrastive_loss * 10.0

    """ box mask constraints was proposed in HiVG """
    if args.use_box_mask_constraints or args.enable_dynamic_mim:
        coef_focal = 20.0
        coef_dice = 2.0
        patch_num = int(mpmath.sqrt(visu_sim.shape[-1]))
        # Downward interpolation, the shape of tgt_mask is B C H W.
        obj_mask = mdetr_interpolate(tgt_mask.float(), (patch_num, patch_num), mode="nearest")[:, 0] > 0.5
        obj_mask = obj_mask.flatten(1).float()
        visu_sim = visu_sim.flatten(1)
        losses['loss_mrm_focal'] = sigmoid_focal_loss(visu_sim, obj_mask, num_boxes) * coef_focal
        losses['loss_mrm_dice'] = dice_loss(visu_sim, obj_mask, num_boxes) * coef_dice

    if args.use_mask_loss:
        coef_focal = 20.0
        coef_dice = 2.0
        # Interpolation upwards, the shape of seg_mask is B C H W
        src_mask = mdetr_interpolate(seg_mask, size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False)
        src_mask = src_mask.flatten(1)
        tgt_mask = tgt_mask.flatten(1).float()

        losses['loss_seg_focal'] = sigmoid_focal_loss(src_mask, tgt_mask, num_boxes) * coef_focal
        losses['loss_seg_dice'] = dice_loss(src_mask, tgt_mask, num_boxes) * coef_dice

    if args.enable_ref_mlm and mlm_loss is not None:
        # losses['loss_mlm'] = mlm_loss * 10.0
        losses['loss_mlm'] = mlm_loss
        if args.enable_mlm_sts and mlm_sts_pred.shape == mlm_sts_labels.shape:
            kl_loss = nn.KLDivLoss(reduction="batchmean")  # mlm_sts_pred is torch.Size([64, 62])
            losses['loss_mlm_sts'] = kl_loss(F.log_softmax(mlm_sts_pred, dim=-1), F.softmax(mlm_sts_labels, dim=-1))

    if args.enable_ref_mim and mim_pred is not None:
        loss_fn = nn.CrossEntropyLoss()
        if isinstance(mim_pred, list):
            loss_1 = loss_fn(input=mim_pred[0], target=mim_labels)
            loss_2 = loss_fn(input=mim_pred[1], target=mim_labels)
            losses['loss_mim'] = loss_1 + loss_2
        else:
            # mim pred shape:  torch.Size([5520, 8192]), mim_labels:  torch.Size([5520])
            # The 0-th dimension of min_pred and mim_labels varies due to the random number of mask positions
            losses['loss_mim'] = loss_fn(input=mim_pred, target=mim_labels)  # tensor(9.7763, device='cuda:0')
            if args.enable_mim_vts and mim_vts_pred is not None:
                mim_vts_loss = F.l1_loss(mim_vts_pred, mim_vts_labels, reduction='none')  # torch.Size([64, 576, 4])
                # Implementation version 2
                losses['loss_mim_vts'] = mim_vts_loss.sum(dim=-1).mean()  # tensor(1.7184, device='cuda:0')
                # Implementation version 2
                # losses['loss_mim_vts'] = mim_vts_loss.sum(dim=-1).sum(dim=-1).mean()  # tensor(429.6270, 'cuda:0')

    return losses

