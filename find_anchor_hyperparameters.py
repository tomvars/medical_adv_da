import torch
import logging
import numpy as np
from src.medicaldetectiontoolkit.retina_unet import net as retina_unet
import src.medicaldetectiontoolkit.model_utils as mutils
from src.utils import loop_iterable
from dataclasses import dataclass, field
from src.dataset import get_monai_patch_dataset
from monai.data import DataLoader

def get_3d_retina_unet(spatial_size,
                       base_rpn_anchor_scale_xy=4,
                       base_rpn_anchor_scale_z=2,
                       base_backbone_strides_xy=4,
                       base_backbone_strides_z=1):
    logger = logging.getLogger()
    
    @dataclass
    class Config:
        head_classes: int = 2
        start_filts: int = 48
        end_filts: int = 48*2  # start_filts * 4
        res_architecture: str = 'resnet101'
        sixth_pooling: bool = False
        n_channels: int = 1
        n_latent_dims: int = 0
        num_seg_classes: int = 2
        norm: str = 'instance_norm'
        relu: str = 'leaky_relu'
        n_rpn_features: int = 64 # 128 in 3D
        rpn_anchor_ratios: list = field(default_factory=lambda: [0.5, 1, 2])
        rpn_train_anchors_per_image: int = 5
        anchor_matching_iou: float = 0.3
        roi_chunk_size: int = 600
        n_anchors_per_pos: int = 9 #len(cf.rpn_anchor_ratios) * 3
        rpn_anchor_stride: int = 1
        pre_nms_limit: int = 50000 #3000 if self.dim == 2 else 6000
        rpn_bbox_std_dev: np.array = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        bbox_std_dev: np.array = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        dim: int = 3
        scale: np.array = field(default_factory=lambda: np.array([spatial_size[0], spatial_size[1],
                                                                  spatial_size[0], spatial_size[1],
                                                                  spatial_size[2], spatial_size[2]]))
        window: np.array = field(default_factory=lambda: np.array([0, 0, spatial_size[0],
                                                                   spatial_size[0], 0,
                                                                   spatial_size[2]]))
        detection_nms_threshold: float = 1e-5
        model_max_instances_per_batch_element: int = 30 #10 if self.dim == 2 else 30
        model_min_confidence: float = 0.1
        weight_init: str = None
        patch_size: np.array = field(default_factory=lambda: np.array(spatial_size))
        backbone_path: str = '/home/tom/DomainAdaptationJournal/src/medicaldetectiontoolkit/fpn.py'
        operate_stride1: int = True
        pyramid_levels: list = field(default_factory=lambda: [0, 1, 2, 3])
        rpn_anchor_ratios: list = field(default_factory=lambda: [0.5, 1, 2])
        rpn_anchor_scales: dict = field(default_factory=lambda: {'xy': [[base_rpn_anchor_scale_xy],
                                                                        [base_rpn_anchor_scale_xy*2],
                                                                        [base_rpn_anchor_scale_xy*4],
                                                                        [base_rpn_anchor_scale_xy*8]],
                                                                 'z': [[base_rpn_anchor_scale_z],
                                                                       [base_rpn_anchor_scale_z*2],
                                                                       [base_rpn_anchor_scale_z*4],
                                                                       [base_rpn_anchor_scale_z*8]]})
        backbone_strides: dict = field(default_factory=lambda:  {'xy': [base_backbone_strides_xy,
                                                                        base_backbone_strides_xy*2,
                                                                        base_backbone_strides_xy*4,
                                                                        base_backbone_strides_xy*8],
                                                                 'z': [base_backbone_strides_z,
                                                                       base_backbone_strides_z*2,
                                                                       base_backbone_strides_z*4,
                                                                       base_backbone_strides_z*8]})
    cf = Config()

    cf.backbone_shapes = np.array(
                [[int(np.ceil(cf.patch_size[0] / stride)),
                  int(np.ceil(cf.patch_size[1] / stride)),
                  int(np.ceil(cf.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(cf.backbone_strides['xy'], cf.backbone_strides['z']
                                             )])
    cf.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            cf.rpn_anchor_scales['xy']]
    cf.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                   cf.rpn_anchor_scales['z']]
    cf.operate_stride1 = True
    print(cf.rpn_anchor_scales)
    print(cf.backbone_strides)
    return retina_unet(cf=cf, logger=logger)

def simple_anchor_matching(cf, anchors, gt_boxes, gt_class_ids):
    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2*cf.dim))
    anchor_matching_iou = cf.anchor_matching_iou

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = mutils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    return anchor_iou_max.max(), overlaps

dataset_3d = get_monai_patch_dataset('/data2/tom/microbleeds/EPAD_plus_rCMB_manual/', 128, 'mask',
                        "dataset_split_epad_rcmb_manual_combined.csv", 'train',
                        spatial_size=[128, 128, 24], spatial_dims=3, exclude_slices=None,
                        synthesis=True, bounding_boxes=True,
                        return_aug=False, label_mapping= {0: 0, 1: 1, 2: 1, 3:1})
source_dl = loop_iterable(DataLoader(dataset_3d, batch_size=10, shuffle=True, collate_fn=lambda x: x, drop_last=True))

def run_expt(spatial_size=[128, 128, 24],
             base_rpn_anchor_scale_xy=4,
             base_rpn_anchor_scale_z=2,
             base_backbone_strides_xy=4,
             base_backbone_strides_z=1):
    with torch.no_grad():
        retina_unet_test = get_3d_retina_unet(spatial_size=spatial_size,
                                              base_rpn_anchor_scale_xy=base_rpn_anchor_scale_xy,
                                              base_rpn_anchor_scale_z=base_rpn_anchor_scale_z,
                                              base_backbone_strides_xy=base_backbone_strides_xy,
                                              base_backbone_strides_z=base_backbone_strides_z)
        anchor_maxes = []
        for idx in range(10):
            source_batch = next(source_dl)
            gt_class_ids = [f[0]['labels_class_target'] for f in source_batch]
            gt_boxes = [f[0]['labels_bbox'] for f in source_batch]
            source_inputs, source_labels = (np.stack([f[0]['inputs'] for f in source_batch]),
                                                    np.stack([f[0]['labels'] for f in source_batch]))
            source_inputs = torch.from_numpy(source_inputs).float().cuda()
            if idx == 0:
                detections, class_logits, pred_deltas, seg_logits, fpn_outs = retina_unet_test(source_inputs)
                print('Dimensions check passed!')

            for b in range(len(gt_boxes)):
                if gt_boxes[b].size:
                    anchor_iou_max, overlaps = simple_anchor_matching(
                        retina_unet_test.cf, retina_unet_test.np_anchors, gt_boxes[b], gt_class_ids[b])
                    anchor_maxes.append(anchor_iou_max)
                    print(retina_unet_test.np_anchors[np.argmax(overlaps)])
                    print(gt_boxes[b])
                    print(anchor_iou_max)
        anchor_maxes = np.array(anchor_maxes)
        def volume(coords):
            return (coords[3] - coords[1]) * (coords[2] - coords[0]) * (coords[5] - coords[4]) 
        print(np.array([volume(x) for x in  retina_unet_test.np_anchors]).mean())
        return anchor_maxes.mean(), anchor_maxes.std()

# print(run_expt(base_rpn_anchor_scale_z=3, base_rpn_anchor_scale_xy=3))
print(run_expt(base_rpn_anchor_scale_z=2, base_rpn_anchor_scale_xy=2))
# print(run_expt(base_rpn_anchor_scale_z=2, base_rpn_anchor_scale_xy=2))
# print(run_expt(base_rpn_anchor_scale_xy=1, base_rpn_anchor_scale_z=1))
# print(run_expt(base_rpn_anchor_scale_xy=4, base_rpn_anchor_scale_z=4))

# print(run_expt(base_rpn_anchor_scale_xy=2))
# print(run_expt(base_rpn_anchor_scale_xy=1))

