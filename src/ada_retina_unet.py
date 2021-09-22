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
from src.networks import get_retina_unet
from src.utils import save_bboxes_for_plotting
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform
import src.medicaldetectiontoolkit.model_utils as mutils
from src.medicaldetectiontoolkit.retina_unet import compute_class_loss, compute_bbox_loss, get_results

class AdaRetinaUNetModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.retina_unet = get_retina_unet()
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.retina_unet.parameters(), lr=self.cf.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[20000, 20000], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
    
    def initialise(self):
        self.p = multiprocessing.Pool(10)
        # Gradient clipping!
        torch.nn.utils.clip_grad_norm(itertools.chain(self.retina_unet.parameters()), 1.0)
    
    def inner_training_loop(self, batch, inputs, labels):
        img = inputs
        gt_class_ids = [f['labels_class_target'] for f in batch]
        gt_boxes = [f['labels_bbox'] for f in batch]
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
                anchor_class_match, anchor_target_deltas = mutils.gt_anchor_matching(
                    self.retina_unet.cf, self.retina_unet.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.retina_unet.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

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
        seg_loss_dice = 1 - mutils.batch_dice(F.softmax(seg_logits, dim=1),var_seg_ohe)
        seg_loss_ce = F.cross_entropy(seg_logits, var_seg[:, 0])
        loss = batch_class_loss + batch_bbox_loss + (seg_loss_dice + seg_loss_ce) / 2
        return loss, results_dict, seg_loss_dice, seg_loss_ce, batch_clas_loss, batch_bbox_loss, seg_logits
        
        
    def training_loop(self, source_dl, target_dl):
        self.scheduler.step()
        postfix_dict, tensorboard_dict = {}, {}
        # Data is coming in in un-collated batches, need to collate them here 
        source_batch = next(source_dl)
        source_inputs, source_labels = (np.stack([f['inputs'] for f in source_batch]),
                                        np.stack([f['labels'] for f in source_batch]))
        target_batch = next(target_dl)
        target_inputs, target_labels = (np.stack([f['inputs'] for f in target_batch]),
                                        np.stack([f['labels'] for f in target_batch]))
        
        # This could be a function where we input source_batch, source_labels and output 
        # results_dict, seg_loss_dice, seg_loss_ce, batch_clas_loss, batch_bbox_loss, seg_logits
        loss, results_dict, seg_loss_dice, seg_loss_ce, batch_clas_loss, batch_bbox_loss, seg_logits = self.inner_training_loop(
            source_batch,
            source_inputs,
            source_labels
        )
        postfix_dict['torch_loss'] = loss.item()
        postfix_dict['class_loss'] = batch_class_loss.item()
        postfix_dict['bbox_loss'] = batch_bbox_loss.item()
        postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
        postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
        postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
        tensorboard_dict = {
            'source_inputs': source_inputs,
            'source_labels': source_labels,
            'outputs_source': F.softmax(seg_logits, dim=1),
            'results_dict': results_dict
        }
        loss.backward()
        self.seg_optimizer.step()
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        postfix_dict, tensorboard_dict = {}, {}
        with torch.no_grad():
            # Data is coming in in un-collated batches, need to collate them here 
            source_batch = next(source_dl)
            source_inputs, source_labels = (np.stack([f['inputs'] for f in source_batch]),
                                            np.stack([f['labels'] for f in source_batch]))
    #         target_batch = next(target_dl)
    #         target_inputs, target_labels = (torch.tensor(np.stack([f['inputs'] for f in target_batch])).to(self.device),
    #                                         torch.tensor(np.stack([f['labels'] for f in target_batch])).to(self.device))

            # Training segmentation model from Generated T2s
            img = source_inputs
            gt_class_ids = [f['labels_class_target'] for f in source_batch]
            gt_boxes = [f['labels_bbox'] for f in source_batch]
            var_seg_ohe = torch.FloatTensor(mutils.get_one_hot_encoding(source_labels, self.cf.labels)).cuda()
            var_seg = torch.LongTensor(source_labels).cuda()

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
                    anchor_class_match, anchor_target_deltas = mutils.gt_anchor_matching(
                        self.retina_unet.cf, self.retina_unet.np_anchors, gt_boxes[b], gt_class_ids[b])

                    # add positive anchors used for loss to results_dict for monitoring.
                    pos_anchors = mutils.clip_boxes_numpy(
                        self.retina_unet.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                    for p in pos_anchors:
                        box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

                else:
                    anchor_class_match = np.array([-1]*self.retina_unet.np_anchors.shape[0])
                    anchor_target_deltas = np.array([0])

                anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
                anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

                # compute losses.
                class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
                bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

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
            postfix_dict['torch_loss'] = loss.item()
            postfix_dict['class_loss'] = batch_class_loss.item()
            postfix_dict['bbox_loss'] = batch_bbox_loss.item()
            postfix_dict['seg_loss_dice'] =  seg_loss_dice.item()
            postfix_dict['seg_loss_ce'] =  seg_loss_ce.item()
            postfix_dict['mean_seg_preds'] = np.mean(results_dict['seg_preds'])
            tensorboard_dict = {
                'source_inputs': img,
                'source_labels': var_seg,
                'outputs_source': F.softmax(seg_logits, dim=1),
                'results_dict': results_dict,
            }
        return postfix_dict, tensorboard_dict
    
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
        elif self.cf.data_task == 'ms':
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='source_labels', png=True)
            save_images(writer=self.writer, images=tensorboard_dict['target_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='target_labels', png=True)
            save_images(writer=self.writer, images=tensorboard_dict['outputs'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='outputs_source', png=True)
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                        sigmoid=False, png=True,
                        iteration=self.iterations, name='source_inputs')
            save_images(writer=self.writer, images=tensorboard_dict['target_inputs'], normalize=True,
                        sigmoid=False, png=True,
                        iteration=self.iterations, name='targets_inputs')
        elif self.cf.data_task == 'crossmoda':
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                                   sigmoid=False, png=False,
                                   iteration=self.iterations, name='source_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['outputs_source'], normalize=False, sigmoid=True,
                                   iteration=self.iterations, name='outputs_source_cochlea/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                                   iteration=self.iterations, name='cochlea_labels/{}'.format(split), png=False)
            save_bboxes_for_plotting(writer=self.writer, img=tensorboard_dict['source_inputs'], name='bbox_plot/{}'.format(split),
                                     results_dict=tensorboard_dict['results_dict'], iteration=self.iterations)
            for key, value in postfix_dict.items():
                self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)            

                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.retina_unet = self.retina_unet.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.retina_unet.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
