import multiprocessing
import sys
import os
sys.path.append('../')
import torch
import torch.optim as optim
import numpy as np
from functools import partial
from networks import Destilation_student_matchingInstance
from utils import save_images
from utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform

class SupervisedJointModel:
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.seg_model = Destilation_student_matchingInstance(self.cf.labels - 1, self.cf.channels)
        self.seg_model.cuda()
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.cf.lr)
        step_1 = 20000 if self.cf.task == 'ms' else 5000
        step_2 = 20000 if self.cf.task == 'ms' else 10000
        self.scheduler = optim.lr_scheduler.MultiStepLR(seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
    
    def initialise(self):
        self.seg_model.cuda()
        self.p = multiprocessing.Pool(10)
        
    def training_loop(self, source_dl, target_dl):
        self.scheduler.step()
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        batch_source = source_inputs.cpu().numpy()
        batch_source = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_source))
        batch_source = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_source))
        inputstaug = torch.Tensor(batch_source).cuda()
        theta_source, Theta_inv = generate_affine(inputstaug,
                                                  degreeFreedom=self.cf.affine_rot_degree,
                                                  scale=self.cf.affine_scale,
                                                  shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, theta_source)
        outputs, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        batch_target = target_inputs.cpu().numpy()
        batch_target = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_target))
        batch_target = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_target))
        inputstaug = torch.Tensor(batch_target).cuda()
        theta_target, Theta_inv = generate_affine(inputstaug,
                                                  degreeFreedom=self.cf.affine_rot_degree,
                                                  scale=self.cf.affine_scale,
                                                  shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, theta_target)
        outputst, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        source_labels_transformed = apply_transform(source_labels, theta_source)
        target_labels_transformed = apply_transform(target_labels, theta_target)
        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels_transformed) +\
                          dice_soft_loss(torch.sigmoid(outputst), target_labels_transformed)
        loss = supervised_loss
        self.seg_model.zero_grad()
        loss.backward()
        self.seg_optimizer.step()
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'source_labels': source_labels,
                            'target_labels': target_labels,
                            'inputstaug': inputstaug,
                            'outputs': outputs,
                            'outputst': outputst}
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        batch_source = source_inputs.cpu().numpy()
        batch_source = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_source))
        batch_source = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_source))
        inputstaug = torch.Tensor(batch_source).cuda()
        theta_source, Theta_inv = generate_affine(inputstaug,
                                                  degreeFreedom=self.cf.affine_rot_degree,
                                                  scale=self.cf.affine_scale,
                                                  shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, theta_source)
        outputs, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        batch_target = target_inputs.cpu().numpy()
        batch_target = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_target))
        batch_target = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_target))
        inputstaug = torch.Tensor(batch_target).cuda()
        theta_target, Theta_inv = generate_affine(inputstaug,
                                                  degreeFreedom=self.cf.affine_rot_degree,
                                                  scale=self.cf.affine_scale,
                                                  shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, theta_target)
        outputst, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        source_labels_transformed = apply_transform(source_labels, theta_source)
        target_labels_transformed = apply_transform(target_labels, theta_target)
        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels_transformed) +\
                          dice_soft_loss(torch.sigmoid(outputst), target_labels_transformed)
        loss = supervised_loss
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'source_labels': source_labels,
                            'target_labels': target_labels,
                            'inputstaug': inputstaug,
                            'outputs': outputs,
                            'outputst': outputst}
        return postfix_dict, tensorboard_dict
        
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.task == 'tumour':
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
        elif self.cf.task == 'ms':
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
            if inputstaug is not None:
                save_images(writer=self.writer, images=tensorboard_dict['inputstaug'], normalize=True, sigmoid=False,
                            iteration=self.iterations, name='inputsaug')
            if outputst is not None:
                save_images(writer=self.writer, images=tensorboard_dict['outputst'], normalize=False, sigmoid=True,
                            iteration=self.iterations, name='outputs_target')
                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.seg_model = self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
