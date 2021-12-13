# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools
from functools import partial
from src.base_model import BaseModel
from src.networks import get_3d_retina_unet
from src.utils import save_bboxes_for_plotting
from src.utils import create_overlap_image
from src.utils import create_bbox_overlap_volume
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform
import src.medicaldetectiontoolkit.model_utils as mutils
from src.medicaldetectiontoolkit.retina_unet import compute_class_loss, compute_bbox_loss, get_results
from src.atss_matcher import gt_anchor_matching_atss, compute_giou_loss

anchor_params_dict = {
    'microbleed': (2, 2, 4, 1),
    'crossmoda': (4, 4, 4, 1),
    'ms': (4, 4, 4, 1)
}

class SupervisedFCOS3DModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        """
        For each pixel inside a bounding box we predict the displacement in 3D to reach the edges.
        W
        """
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.retina_unet = get_3d_retina_unet(self.cf.spatial_size,
                                              base_rpn_anchor_scale_xy=anchor_params_dict[cf.data_task][0],
                                              base_rpn_anchor_scale_z=anchor_params_dict[cf.data_task][1],
                                              base_backbone_strides_xy=anchor_params_dict[cf.data_task][2],
                                              base_backbone_strides_z=anchor_params_dict[cf.data_task][3])
        self.writer = writer
        self.seg_optimizer = optim.SGD(self.retina_unet.parameters(), lr=self.cf.lr, weight_decay=1e-5, nesterov=True, momentum=0.9)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[20000, 20000], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
        if cf.anchor_matching_strategy == "atss":
            self.anchor_matcher = gt_anchor_matching_atss
        elif cf.anchor_matching_strategy == "iou":
            self.anchor_matcher = mutils.gt_anchor_matching
        self.bbox_loss = cf.bbox_loss
    
    def initialise(self):
        pass
        
    def training_loop(self, source_dl, target_dl):
        self.scheduler.step()
        self.retina_unet.train()
        postfix_dict, tensorboard_dict = {}, {}
        # Data is coming in in un-collated batches, need to collate them here 
        source_batch = next(source_dl)
        source_inputs, source_labels = (np.stack([f[0]['inputs'] for f in source_batch]),
                                        np.stack([f[0]['labels'] for f in source_batch]))
        
        # Training segmentation model from Generated T2s
        img = source_inputs
        gt_class_ids = [f[0]['labels_class_target'] for f in source_batch]
        gt_boxes = [f[0]['labels_bbox'] for f in source_batch]
        var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(source_labels, self.cf.labels)).cuda()
        var_seg = torch.LongTensor(source_labels).cuda()

        img = torch.from_numpy(img).float().cuda()
        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()
        batch_non_zero_elements = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas, seg_logits, fpn_outs = self.retina_unet(img)
        # loop over batch
        
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': gt_boxes[b][ix],
                                                'box_label': gt_class_ids[b][ix], 'box_type': 'gt'})
                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas, anchor_class_matches_w_gt_idx = self.anchor_matcher(
                    self.retina_unet.cf, self.retina_unet.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.retina_unet.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])
                anchor_class_matches_w_gt_idx = np.array([-1]*self.retina_unet.np_anchors.shape[0])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_class_matches_w_gt_idx = torch.from_numpy(anchor_class_matches_w_gt_idx).cuda().long()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            assert anchor_class_match.shape[0] == class_logits[b].shape[0]
            pos_indices = torch.nonzero(anchor_class_match > 0)
            # TODO: Investigate class_logits, is it getting differentiated? check weights of Classifier.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            if self.bbox_loss == 'l1':
                bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)
            elif self.bbox_loss == 'giou':
                #Tom: We've lost the connection between anchor and associated gt... only
                #anchor_target_deltas, pred_deltas, anchor_class_match, gt_boxes, anchor_class_match_w_idx
                bbox_loss = compute_giou_loss(pred_deltas=pred_deltas[b],
                                              anchor_class_match=anchor_class_match,
                                              gt_boxes=gt_boxes[b],
                                              anchor_class_match_w_idx=anchor_class_matches_w_gt_idx,
                                              anchors=self.retina_unet.anchors)

            # add negative anchors used for loss to results_dict for monitoring.
            neg_anchors = mutils.clip_boxes_numpy(
                self.retina_unet.np_anchors[np.argwhere(anchor_class_match.cpu().numpy() == -1)][neg_anchor_ix, 0],
                img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss
            if len(gt_boxes[b]) > 0:
                batch_non_zero_elements += 1
        if batch_non_zero_elements > 0:
            batch_bbox_loss /=  batch_non_zero_elements
    
        results_dict = get_results(self.retina_unet.cf, img.shape, detections, seg_logits, box_results_list)
        results_dict['box_results_list'] = box_results_list
        if torch.isnan(seg_logits).any():
            #'/data2/tom/microbleeds/VALDO_T2S/slices/flair/FLAIR_sub-227_slice_0047.nii.gz'
            exit()
        seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1), var_seg_ohe)
        seg_loss_ce = (F.cross_entropy(seg_logits, var_seg[:, 0]))
        loss =  batch_class_loss + batch_bbox_loss + (seg_loss_dice + seg_loss_ce) / 2
#         loss =  batch_class_loss
        self.seg_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(itertools.chain(self.retina_unet.parameters()), 10.0)
        loss.backward()
        self.seg_optimizer.step()
        postfix_dict['torch_loss'] = loss.item()
        postfix_dict['class_loss'] = batch_class_loss.item()
        postfix_dict['bbox_loss'] = batch_bbox_loss.item()
        postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
#         postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
        postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
        tensorboard_dict = {
            'source_inputs': img,
            'source_labels': var_seg,
            'source_outputs': F.softmax(seg_logits, dim=1)[:, 1:, ...],
            'results_dict': results_dict
        }
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        postfix_dict, tensorboard_dict = {}, {}
        with torch.no_grad():
            self.retina_unet.eval()
            # Data is coming in in un-collated batches, need to collate them here 
            
            source_batch = next(source_dl)
            source_inputs, source_labels = (np.stack([f[0]['inputs'] for f in source_batch]),
                                            np.stack([f[0]['labels'] for f in source_batch]))

            # Training segmentation model from Generated T2s
            img = source_inputs
            gt_class_ids = [f[0]['labels_class_target'] for f in source_batch]
            gt_boxes = [f[0]['labels_bbox'] for f in source_batch]
            var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(source_labels, self.cf.labels)).cuda()
            var_seg = torch.LongTensor(source_labels).cuda()

            img = torch.from_numpy(img).float().cuda()
            
            # For batch_class_loss and batch_class_loss, need to take average over non-zero values.
            batch_class_loss = torch.FloatTensor([0]).cuda()
            batch_bbox_loss = torch.FloatTensor([0]).cuda()
            batch_non_zero_elements = torch.FloatTensor([0]).cuda()

            # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
            box_results_list = [[] for _ in range(img.shape[0])]
            detections, class_logits, pred_deltas, seg_logits, fpn_outs = self.retina_unet(img)
            # loop over batch
            for b in range(img.shape[0]):

                # add gt boxes to results dict for monitoring.
                if len(gt_boxes[b]) > 0:
                    for ix in range(len(gt_boxes[b])):
                        box_results_list[b].append({'box_coords': gt_boxes[b][ix],
                                                    'box_label': gt_class_ids[b][ix], 'box_type': 'gt'})
                   # match gt boxes with anchors to generate targets.
                    anchor_class_match, anchor_target_deltas, anchor_class_matches_w_gt_idx = self.anchor_matcher(
                        self.retina_unet.cf, self.retina_unet.np_anchors, gt_boxes[b], gt_class_ids[b])

                    # add positive anchors used for loss to results_dict for monitoring.
                    pos_anchors = mutils.clip_boxes_numpy(
                        self.retina_unet.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                    for p in pos_anchors:
                        box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

                else:
                    anchor_class_match = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                    anchor_target_deltas = np.array([0])
                    anchor_class_matches_w_gt_idx = np.array([-1]*self.retina_unet.np_anchors.shape[0])

                anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
                anchor_class_matches_w_gt_idx = torch.from_numpy(anchor_class_matches_w_gt_idx).cuda().long()
                anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

                # compute losses.
                class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
                if self.bbox_loss == 'l1':
                    bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)
                elif self.bbox_loss == 'giou':
                    #Tom: We've lost the connection between anchor and associated gt... only
                    #anchor_target_deltas, pred_deltas, anchor_class_match, gt_boxes, anchor_class_match_w_idx
                    bbox_loss = compute_giou_loss(pred_deltas=pred_deltas[b],
                                                  anchor_class_match=anchor_class_match,
                                                  gt_boxes=gt_boxes[b],
                                                  anchor_class_match_w_idx=anchor_class_matches_w_gt_idx,
                                                  anchors=self.retina_unet.anchors)

                # add negative anchors used for loss to results_dict for monitoring.
                neg_anchors = mutils.clip_boxes_numpy(
                    self.retina_unet.np_anchors[np.argwhere(anchor_class_match.cpu().numpy() == -1)][neg_anchor_ix, 0],
                    img.shape[2:])
                for n in neg_anchors:
                    box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})
                
                batch_class_loss += class_loss / img.shape[0]
                batch_bbox_loss += bbox_loss
                if len(gt_boxes[b]) > 0:
                    batch_non_zero_elements += 1
            if batch_non_zero_elements > 0:
                batch_bbox_loss /=  batch_non_zero_elements
                
            results_dict = get_results(self.retina_unet.cf, img.shape, detections, seg_logits, box_results_list)
            results_dict['box_results_list'] = box_results_list
            if torch.isnan(seg_logits).any():
                #'/data2/tom/microbleeds/VALDO_T2S/slices/flair/FLAIR_sub-227_slice_0047.nii.gz'
                exit()
            seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1), var_seg_ohe)
            seg_loss_ce = (F.cross_entropy(seg_logits, var_seg[:, 0]))
            loss = (batch_class_loss + batch_bbox_loss) + (seg_loss_dice + seg_loss_ce) / 2
            postfix_dict['torch_loss'] = loss.item()
            postfix_dict['class_loss'] = batch_class_loss.item()
            postfix_dict['bbox_loss'] = batch_bbox_loss.item()
            postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
            postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
            postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
            tensorboard_dict = {
                'source_inputs': img,
                'source_labels': var_seg,
                'source_outputs': F.softmax(seg_logits, dim=1)[:, 1:, ...],
                'results_dict': results_dict
            }
            return postfix_dict, tensorboard_dict
        
    def inference_func(self, inference_inputs, mode='seg_out', bbox_thresh=0.14):
        """
        Expects inputs as torch tensor
        
        mode: if 'seg_out' outputs the segmentation branch output
              if 'box_out' outputs a segmentation mask of detections
        """
        with torch.no_grad():
            self.retina_unet.eval()
            # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
            box_results_list = [[] for _ in range(inference_inputs.shape[0])]
            detections, class_logits, pred_deltas, seg_logits, fpn_outs = self.retina_unet(inference_inputs)
            results_dict = get_results(self.retina_unet.cf, inference_inputs.shape,
                                       detections, seg_logits)
            output_dict = {
                    'source_inputs': inference_inputs,
                    'source_outputs': F.softmax(seg_logits, dim=1)[:, 1:, ...],
                    'results_dict': results_dict
                }
        if mode == 'seg_out':
            return (F.softmax(seg_logits, dim=1)[:, 1:, ...] > 0.5).to(torch.bool)
        elif mode == 'box_out':
            all_bboxes = results_dict['boxes']
            box_out = []
            for single_volume_bboxes in all_bboxes:
                # Size of a single volume
                output_mask = torch.zeros_like(inference_inputs[0])
                for detection_idx, b in enumerate(single_volume_bboxes):
                    if b['box_score'] > bbox_thresh:
                        print(b['box_type'])
                        print(b['box_pred_class_id'])
                        print(b['box_score'])
                        # bbox coords in y1, x1, y2, x2, z1, z2 format
                        bbox_coords = b['box_coords']
                        # Assign each bbox a unique value
                        output_mask[:, bbox_coords[1]:bbox_coords[3],
                                    bbox_coords[0]:bbox_coords[2],
                                    bbox_coords[4]:bbox_coords[5]] = detection_idx + 1
                box_out.append(output_mask)
            return torch.stack(box_out)
        else:
            raise NotImplemented

    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.data_task == 'tumour':
            for idx, modality in enumerate(['flair', 't1c', 't1', 't2']):
                save_images(writer=self.writer, images=tensorboard_dict['source_inputs'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name='source_' + modality)
                save_images(writer=self.writer, images=tensorboard_dict['target_inputs'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name='target_' + modality)
                save_images(writer=self.writer, images=tensorboard_dict['inputstaug'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name=modality + '_aug')
        elif self.cf.data_task in ['microbleed', 'crossmoda']:
            create_bbox_overlap_volume(writer=self.writer,
                                       images=tensorboard_dict['source_inputs'],
                                       results_dict=tensorboard_dict['results_dict'],
                                       iteration=self.iterations,
                                       name='bbox_plot_source/{}'.format(split),
                                       tensorboard=True, png=False)
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['source_inputs'],
                                 preds=tensorboard_dict['source_outputs'],
                                 labels=tensorboard_dict['source_labels'],
                                 iteration=self.iterations, name='source_confusion_plot/{}'.format(split))
            for key, value in postfix_dict.items():
                self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)     

                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
#         pretrained_dict = {k.replace('BBRegressor.', ''): v for k, v in checkpoint['seg_model'].items() if k.startswith('BBRegressor')}
#         self.retina_unet.BBRegressor.load_state_dict(pretrained_dict)
#         pretrained_dict = {k.replace('Fpn.', ''): v for k, v in checkpoint['seg_model'].items() if k.startswith('Fpn')}
#         self.retina_unet.Fpn.load_state_dict(pretrained_dict)
#         pretrained_dict = {k.replace('final_conv.', ''): v for k, v in checkpoint['seg_model'].items() if k.startswith('final_conv')}
#         self.retina_unet.final_conv.load_state_dict(pretrained_dict)
#         print(checkpoint['seg_model'].keys())
        self.retina_unet.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.retina_unet.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
