############################################################################
#
# Philosophy here is to write as little original code as possible, use  
# functions from nnDetection or medicaldetectiontoolkit wherever possible
#
############################################################################

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.core.boxes.coder import decode_single
from nndet.core.boxes.ops import generalized_box_iou
from nndet.core.boxes.sampler import HardNegativeSampler

import numpy as np
import torch
import pdb


atss_matcher = ATSSMatcher(num_candidates=4, center_in_gt=False)
fg_bg_sampler = HardNegativeSampler(batch_size_per_image=2,
                                    positive_fraction=0.333,
                                    min_neg=2,
                                    pool_size=10)

def convert_coordinates(bboxes):
    """
    Args: bboxes -> [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    Returns: bboxes -> [num_anchors, (x1, y1, x2, y2, (z1, z2))]
    """
    if bboxes.shape[1] == 6:
        bboxes[:, np.array([1, 0, 4, 3, 2, 5])]
    elif bboxes.shape[1] == 4:
        bboxes[:, np.array([1, 0, 3, 2])]
    return torch.tensor(bboxes)

def convert_deltas(deltas, gt_bboxes):
    """
    This function takes in bbox deltas and the associated gt bboxes
    and returns the bbox coordinates in nnDetection format
    
    nnDetection format for deltas is (dx, dy, log(dw), log(dh), dz, log(dd))
    
    Args: deltas -> [num_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))]
          gt_bboxes -> [num_gt_bboxes, (y1, x1, y2, x2, (z1), (z2))]
    Returns: pred_bboxes -> [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    """
    if bboxes.shape[1] == 6:
        gt_bboxes = gt_bboxes[:, np.array([1, 0, 4, 3, 2, 5])]
        deltas = deltas[:, np.array([1, 0, 4, 3, 2, 5])]
        weights = [1.0] * 6
    elif bboxes.shape[1] == 4:
        gt_bboxes = gt_bboxes[:, np.array([1, 0, 3, 2])]
        deltas = deltas[:, np.array([1, 0, 3, 2])]
        weights = [1.0] * 4
    pred_bboxes = decode_single(rel_codes=deltas,
                                boxes=gt_bboxes,
                                weights=weights,
                                bbox_xform_clip=math.log(1000. / 16)) # Got this from torchvision
    return pred_bboxes

def compute_giou_loss(anchor_target_deltas, pred_deltas, anchor_class_match, gt_boxes):
    """
    Computes the generalised IoU loss between predicted boxes and target boxes
    
    """
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)
    # Here we do hard negative mining!
    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(target_labels, boxes_max_fg_probs)
    pred_boxes_sampled = convert_deltas(box_deltas[sampled_pos_inds], gt_boxes)

    target_boxes_sampled = torch.cat(matched_gt_boxes, dim=0)[sampled_pos_inds]
    return torch.diag(generalized_box_iou(pred_boxes,
                                          target_boxes,
                                          eps=1e-6), diagonal=0).sum()
    
    

def gt_anchor_matching_atss(cf, anchors, gt_boxes, gt_class_ids=None):
    """
    Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    
    Uses ATSS from nnDetection

    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): np.array([num_gt_boxes]) Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)

    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral.
               In case of one stage detectors like RetinaNet/RetinaUNet this flag takes
               class_ids as positive anchor values, i.e. values >= 1!
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    """
    # TODO: Convert from medicaldetectiontoolkit format to nnDetection
    anchors_nndet = convert_coordinates(anchors)
    gt_boxes_nndet = convert_coordinates(gt_boxes)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2*cf.dim))
    # Get num_anchors_per_level
    
    expected_anchors = [np.prod(cf.backbone_shapes[ii]) * len(cf.rpn_anchor_ratios) * len(cf.rpn_anchor_scales['xy'][ii]) for ii in cf.pyramid_levels]
    match_quality_matrix, anchor_class_matches = atss_matcher.compute_matches(
        boxes=gt_boxes_nndet, anchors=anchors_nndet,
        num_anchors_per_level=expected_anchors,
        num_anchors_per_loc=27)
    anchor_class_matches[anchor_class_matches >= 0] = 1
    anchor_iou_argmax = np.argmax(match_quality_matrix.numpy(), axis=0)
    # Leave all negative proposals negative now and sample from them in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches.numpy() > 0)[0]
    ix = 0  # index into anchor_delta_targets
    extra = len(ids) - (cf.rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        extra_ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[extra_ids] = 0
    ids = np.where(anchor_class_matches.numpy() > 0)[0]
    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if cf.dim == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]

        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d

            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)
            ]

        # normalize.
        anchor_delta_targets[ix] /= cf.rpn_bbox_std_dev
        ix += 1
    return anchor_class_matches.numpy(), anchor_delta_targets