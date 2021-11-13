import multiprocessing
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools
from functools import partial
from pathlib import Path
from src.base_model import BaseModel
from src.networks import get_3d_retina_unet
from src.utils import apply_affine_to_coords
from src.utils import save_bboxes_for_plotting
from src.utils import create_bbox_overlap_volume
from src.utils import create_overlap_image
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

class AdaRetinaUNet3DModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
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
        self.seg_optimizer = optim.SGD(self.retina_unet.parameters(), lr=self.cf.lr, weight_decay=0.01, nesterov=True, momentum=0.9)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[20000, 20000], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
        self.anchor_matcher = mutils.gt_anchor_matching #gt_anchor_matching_atss
        self.bbox_loss = 'giou' # generalised_iou_bbox_loss
        # Discriminator setup #
        self.discriminator = Discriminator3D(256, 2, self.cf.discriminator_complexity)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
    
    def initialise(self):
        self.p = multiprocessing.Pool(10)
        # Gradient clipping!
        torch.nn.utils.clip_grad_norm(itertools.chain(self.retina_unet.parameters()), 1.0)
    
    def inner_training_loop(self, batch, inputs, labels):
        img = inputs
        gt_class_ids = [f[0]['labels_class_target'] for f in batch]
        gt_boxes = [f[0]['labels_bbox'] for f in batch]
        var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(labels, self.cf.labels)).cuda()
        var_seg = torch.LongTensor(labels).cuda()

        img = torch.from_numpy(img).float().cuda()
        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()

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
            batch_bbox_loss += bbox_loss / img.shape[0]

        results_dict = get_results(self.retina_unet.cf, img.shape, detections, seg_logits, box_results_list)
        results_dict['box_results_list'] = box_results_list
        seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1), var_seg_ohe)
        seg_loss_ce = F.cross_entropy(seg_logits, var_seg[:, 0])
        loss = batch_class_loss + batch_bbox_loss + (seg_loss_dice + seg_loss_ce) / 2
        return (loss, results_dict, seg_loss_dice, seg_loss_ce, batch_class_loss,
                batch_bbox_loss, class_logits, seg_logits, detections, pred_deltas)
        
        
    def training_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights
        self.scheduler.step()
        postfix_dict, tensorboard_dict = {}, {}
        # Data is coming in in un-collated batches, need to collate them here 
        source_batch = next(source_dl)
        source_inputs, source_labels = (np.stack([f[0]['inputs'] for f in source_batch]),
                                        np.stack([f[0]['labels'] for f in source_batch]))
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug, target_labels = (np.stack([f[0]['inputs'] for f in target_batch]),
                                                           np.stack([f[0]['inputs_aug'] for f in target_batch]),
                                                           np.stack([f[0]['labels'] for f in target_batch]))
        affine_mats = [f[0]['inputs_aug_transforms'][0]['extra_info']['affine'] for f in target_batch]

        
        # This could be a function where we input source_batch, source_labels and output 
        # results_dict, seg_loss_dice, seg_loss_ce, batch_clas_loss, batch_bbox_loss, seg_logits
        (loss, results_dict, seg_loss_dice, seg_loss_ce,
         batch_class_loss, batch_bbox_loss, class_logits, seg_logits,
         detections, pred_deltas) = self.inner_training_loop(
            source_batch,
            source_inputs,
            source_labels
        )
        with torch.no_grad():
            (loss_t, results_dict_t, seg_loss_dice_t, seg_loss_ce_t,
             batch_class_loss_t, batch_bbox_loss_t, class_logits_t, seg_logits_t,
             detections_t, pred_deltas_t) = self.inner_training_loop(
                target_batch,
                target_inputs,
                target_labels
            )
        (loss_ta, results_dict_ta, seg_loss_dice_ta, seg_loss_ce_ta,
         batch_class_loss_ta, batch_bbox_loss_ta, class_logits_ta, seg_logits_ta,
         detections_ta, pred_deltas_ta) = self.inner_training_loop(
            target_batch,
            target_inputs_aug,
            target_labels
        )
        
        
        if alpha > 0:
            # Conceptually we are comparing the _t against the _ta.
            # GT is provided by the predictions on _t
            # Preds are all from _ta
            
            #  detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
            # You have gt_boxes, now you need to do anchor matching and loss calculation
            batch_class_loss_ss = torch.FloatTensor([0]).cuda()
            batch_bbox_loss_ss = torch.FloatTensor([0]).cuda()
            for b in range(source_inputs.shape[0]):
                # add gt boxes to results dict for monitoring.
                pseudo_bbox_preds = [bbox['box_coords'] for bbox in results_dict_t['boxes'][b] if 
                                         bbox['box_type'] == 'det' and bbox['box_score'] > 0.5]
                pseudo_class_preds = [bbox['box_pred_class_id'] for bbox in results_dict_t['boxes'][b] if 
                                      bbox['box_type'] == 'det' and bbox['box_score'] > 0.5]
                transformed_bbox_preds = np.array([apply_affine_to_coords(p, affine_mats[b].cpu().numpy())
                                              for p in pseudo_bbox_preds])
                print('transformed_bbox_preds', transformed_bbox_preds)
                # Adding these to results_dict_t as a gt
                #{'box_coords': array([67, 42, 72, 49, 21, 27]), 'box_label': 1, 'box_type': 'gt'}
                for transformed_bbox_pred in transformed_bbox_preds:
                    results_dict_t['box_results_list'][b].append({'box_coords': transformed_bbox_pred, 'box_type': 'gt'})
                if len(transformed_bbox_preds) > 0:
                    # match gt boxes with anchors to generate targets.
                    anchor_class_match_t, anchor_target_deltas_t, anchor_class_matches_w_gt_idx_t = self.anchor_matcher(
                        self.retina_unet.cf, self.retina_unet.np_anchors, transformed_bbox_preds, pseudo_class_preds)
                else:
                    anchor_class_match_t = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                    anchor_target_deltas_t = np.array([0])
                    anchor_class_matches_w_gt_idx = np.array([-1]*self.retina_unet.np_anchors.shape[0])

                anchor_class_match_t = torch.from_numpy(anchor_class_match_t).cuda()
                anchor_target_deltas_t = torch.from_numpy(anchor_target_deltas_t).float().cuda()
                anchor_class_matches_w_gt_idx_t = torch.from_numpy(anchor_class_matches_w_gt_idx_t).cuda().long()
                # compute losses.
                class_loss_ss, neg_anchor_ix = compute_class_loss(anchor_class_match_t, class_logits_ta[b])
                if self.bbox_loss == 'l1':
                    bbox_loss_ss = compute_bbox_loss(anchor_target_deltas_t, pred_deltas_ta[b], anchor_class_match_t)
                elif self.bbox_loss == 'giou':
                    #Tom: We've lost the connection between anchor and associated gt... only
                    #anchor_target_deltas, pred_deltas, anchor_class_match, gt_boxes, anchor_class_match_w_idx
                    bbox_loss_ss = compute_giou_loss(
                        pred_deltas=pred_deltas_ta[b],
                        anchor_class_match=anchor_class_match_t,
                        gt_boxes=transformed_bbox_preds,
                        anchor_class_match_w_idx=anchor_class_matches_w_gt_idx_t,
                        anchors=self.retina_unet.anchors)
                batch_class_loss_ss += class_loss_ss / source_inputs.shape[0]
                batch_bbox_loss_ss += bbox_loss_ss / source_inputs.shape[0]
            loss = loss + alpha * (batch_class_loss_ss + batch_bbox_loss_ss)
            postfix_dict['class_loss_ss'] = batch_class_loss_ss.item() * alpha
            postfix_dict['bbox_loss_ss'] = batch_bbox_loss_ss.item() * alpha
        
        postfix_dict['torch_loss'] = loss.item()
        postfix_dict['class_loss'] = batch_class_loss.item()
        postfix_dict['bbox_loss'] = batch_bbox_loss.item()
        postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
        postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
        postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
        tensorboard_dict = {
            'source_inputs': torch.tensor(source_inputs).cuda(),
            'source_labels': torch.tensor(source_labels).cuda(),
            'target_labels': torch.tensor(target_labels).cuda(),
            'target_inputs': torch.tensor(target_inputs).cuda(),
            'target_inputs_aug': torch.tensor(target_inputs_aug).cuda(),
            'outputs_source': F.softmax(seg_logits, dim=1)[:, 1:, ...],
            'outputs_target': F.softmax(seg_logits_t, dim=1)[:, 1:, ...],
            'outputs_target_aug': F.softmax(seg_logits_ta, dim=1)[:, 1:, ...],
            'results_dict_source': results_dict,
            'results_dict_t': results_dict_t,
            'results_dict_ta': results_dict_ta,
        }
        self.seg_optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm(itertools.chain(self.retina_unet.parameters()), 10.0)
        loss.backward()
        self.seg_optimizer.step()
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights
        postfix_dict, tensorboard_dict = {}, {}
        with torch.no_grad():
            source_batch = next(source_dl)
            source_inputs, source_labels = (np.stack([f[0]['inputs'] for f in source_batch]),
                                            np.stack([f[0]['labels'] for f in source_batch]))
            target_batch = next(target_dl)
            target_inputs, target_inputs_aug, target_labels = (np.stack([f[0]['inputs'] for f in target_batch]),
                                                               np.stack([f[0]['inputs_aug'] for f in target_batch]),
                                                               np.stack([f[0]['labels'] for f in target_batch]))
            affine_mats = [f[0]['inputs_aug_transforms'][0]['extra_info']['affine'] for f in target_batch]


            # This could be a function where we input source_batch, source_labels and output 
            # results_dict, seg_loss_dice, seg_loss_ce, batch_clas_loss, batch_bbox_loss, seg_logits
            (loss, results_dict, seg_loss_dice, seg_loss_ce,
             batch_class_loss, batch_bbox_loss, class_logits, seg_logits,
             detections, pred_deltas) = self.inner_training_loop(
                source_batch,
                source_inputs,
                source_labels
            )
            (loss_t, results_dict_t, seg_loss_dice_t, seg_loss_ce_t,
             batch_class_loss_t, batch_bbox_loss_t, class_logits_t, seg_logits_t,
             detections_t, pred_deltas_t) = self.inner_training_loop(
                target_batch,
                target_inputs,
                target_labels
            )
            (loss_ta, results_dict_ta, seg_loss_dice_ta, seg_loss_ce_ta,
             batch_class_loss_ta, batch_bbox_loss_ta, class_logits_ta, seg_logits_ta,
             detections_ta, pred_deltas_ta) = self.inner_training_loop(
                target_batch,
                target_inputs_aug,
                target_labels
            )

            if alpha > 0:
                # Conceptually we are comparing the _t against the _ta.
                # GT is provided by the predictions on _t
                # Preds are all from _ta

                #  detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
                # You have gt_boxes, now you need to do anchor matching and loss calculation
                batch_class_loss_ss = torch.FloatTensor([0]).cuda()
                batch_bbox_loss_ss = torch.FloatTensor([0]).cuda()
                for b in range(source_inputs.shape[0]):
                    # add gt boxes to results dict for monitoring.
                    pseudo_bbox_preds = [bbox['box_coords'] for bbox in results_dict_t['boxes'][b] if 
                                         bbox['box_type'] == 'det' and bbox['box_score'] > 0.5]
                    pseudo_class_preds = [bbox['box_pred_class_id'] for bbox in results_dict_t['boxes'][b] if 
                                          bbox['box_type'] == 'det' and bbox['box_score'] > 0.5]
                    transformed_bbox_preds = np.array([apply_affine_to_coords(p, affine_mats[b].cpu().numpy())
                                              for p in pseudo_bbox_preds])
                    print('transformed_bbox_preds', transformed_bbox_preds)
                    if len(transformed_bbox_preds) > 0:
                        # match gt boxes with anchors to generate targets.
                        anchor_class_match_t, anchor_target_deltas_t, anchor_class_matches_w_gt_idx_t = self.anchor_matcher(
                            self.retina_unet.cf, self.retina_unet.np_anchors, transformed_bbox_preds, pseudo_class_preds)
                    else:
                        anchor_class_match_t = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                        anchor_target_deltas_t = np.array([0])
                        anchor_class_matches_w_gt_idx = np.array([-1]*self.retina_unet.np_anchors.shape[0])

                    anchor_class_match_t = torch.from_numpy(anchor_class_match_t).cuda()
                    anchor_target_deltas_t = torch.from_numpy(anchor_target_deltas_t).float().cuda()
                    anchor_class_matches_w_gt_idx_t = torch.from_numpy(anchor_class_matches_w_gt_idx_t).cuda().long()
                    # compute losses.
                    class_loss_ss, neg_anchor_ix = compute_class_loss(anchor_class_match_t, class_logits_ta[b])
                    if self.bbox_loss == 'l1':
                        bbox_loss_ss = compute_bbox_loss(anchor_target_deltas_t, pred_deltas_ta[b], anchor_class_match_t)
                    elif self.bbox_loss == 'giou':
                        #Tom: We've lost the connection between anchor and associated gt... only
                        #anchor_target_deltas, pred_deltas, anchor_class_match, gt_boxes, anchor_class_match_w_idx
                        bbox_loss_ss = compute_giou_loss(
                            pred_deltas=pred_deltas_ta[b],
                            anchor_class_match=anchor_class_match_t,
                            gt_boxes=transformed_bbox_preds,
                            anchor_class_match_w_idx=anchor_class_matches_w_gt_idx_t,
                            anchors=self.retina_unet.anchors)
                    batch_class_loss_ss += class_loss_ss / source_inputs.shape[0]
                    batch_bbox_loss_ss += bbox_loss_ss / source_inputs.shape[0]
                loss = loss + alpha * (batch_class_loss_ss + batch_bbox_loss_ss)
                postfix_dict['class_loss_ss'] = batch_class_loss_ss.item() * alpha
                postfix_dict['bbox_loss_ss'] = batch_bbox_loss_ss.item() * alpha

            postfix_dict['torch_loss'] = loss.item()
            postfix_dict['class_loss'] = batch_class_loss.item()
            postfix_dict['bbox_loss'] = batch_bbox_loss.item()
            postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
            postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
            postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
            tensorboard_dict = {
                'source_inputs': torch.tensor(source_inputs).cuda(),
                'source_labels': torch.tensor(source_labels).cuda(),
                'target_labels': torch.tensor(target_labels).cuda(),
                'target_inputs': torch.tensor(target_inputs).cuda(),
                'target_inputs_aug': torch.tensor(target_inputs_aug).cuda(),
                'outputs_source': F.softmax(seg_logits, dim=1)[:, 1:, ...],
                'outputs_target': F.softmax(seg_logits_t, dim=1)[:, 1:, ...],
                'outputs_target_aug': F.softmax(seg_logits_ta, dim=1)[:, 1:, ...],
                'results_dict_source': results_dict,
                'results_dict_t': results_dict_t,
                'results_dict_ta': results_dict_ta,
            }
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.data_task in ['microbleed', 'crossmoda']:
            create_bbox_overlap_volume(writer=self.writer,
                                       images=tensorboard_dict['source_inputs'],
                                       results_dict=tensorboard_dict['results_dict_source'],
                                       iteration=self.iterations,
                                       name='bbox_plot_source/{}'.format(split),
                                       tensorboard=True, png=False)
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['source_inputs'],
                                 preds=tensorboard_dict['outputs_source'],
                                 labels=tensorboard_dict['source_labels'],
                                 iteration=self.iterations, name='source_confusion_plot/{}'.format(split))
            create_bbox_overlap_volume(writer=self.writer,
                                       images=tensorboard_dict['target_inputs'],
                                       results_dict=tensorboard_dict['results_dict_t'],
                                       iteration=self.iterations,
                                       name='bbox_plot_target/{}'.format(split),
                                       tensorboard=True, png=False)
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['target_inputs'],
                                 preds=tensorboard_dict['outputs_target'],
                                 labels=tensorboard_dict['target_labels'],
                                 iteration=self.iterations, name='target_confusion_plot/{}'.format(split))
            for key, value in postfix_dict.items():
                self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)           

                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.retina_unet.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.retina_unet.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))