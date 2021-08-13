import multiprocessing
import os
import sys
import torch
import torch.optim as optim
import numpy as np
from functools import partial
from src.base_model import BaseModel
from src.networks import Destilation_student_matchingInstance, SplitHeadModel
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform

class SupervisedModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.seg_model = SplitHeadModel(self.cf.channels)
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.cf.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[20000, 20000], gamma=0.1)
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
        # Training segmentation model from Generated T2s
        outputs, outputs2, _, _, _, _, _, _, _, _, _  = self.seg_model(source_inputs)
        tumour_labels = torch.where(source_labels == 1,
                                    torch.ones_like(source_labels),
                                    torch.zeros_like(source_labels))
        cochlea_labels = torch.where(source_labels == 2,
                                     torch.ones_like(source_labels),
                                     torch.zeros_like(source_labels))
        tumour_loss = dice_soft_loss(torch.sigmoid(outputs), tumour_labels)
        cochlea_loss = dice_soft_loss(torch.sigmoid(outputs2), cochlea_labels)
        supervised_loss = (tumour_loss + cochlea_loss) / 2.0
        self.seg_model.zero_grad()
        supervised_loss.backward()
        self.seg_optimizer.step()
        postfix_dict = {'tumour_loss': tumour_loss.item(),
                        'cochlea_loss': cochlea_loss.item(),
                        'supervised_loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'tumour_labels': tumour_labels,
                            'cochlea_labels': cochlea_labels,
                            'outputs_source_tumour': outputs,
                            'outputs_source_cochlea': outputs2}
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        outputs, outputs2, _, _, _, _, _, _, _, _, _  = self.seg_model(source_inputs)
        tumour_labels = torch.where(source_labels == 1,
                                    torch.ones_like(source_labels),
                                    torch.zeros_like(source_labels))
        cochlea_labels = torch.where(source_labels == 2,
                                     torch.ones_like(source_labels),
                                     torch.zeros_like(source_labels))
        tumour_loss = dice_soft_loss(torch.sigmoid(outputs), tumour_labels)
        cochlea_loss = dice_soft_loss(torch.sigmoid(outputs2), cochlea_labels)
        supervised_loss = (tumour_loss + cochlea_loss) / 2.0
        postfix_dict = {'tumour_loss': tumour_loss.item(),
                        'cochlea_loss': cochlea_loss.item(),
                        'supervised_loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'tumour_labels': tumour_labels,
                            'cochlea_labels': cochlea_labels,
                            'outputs_source_tumour': outputs,
                            'outputs_source_cochlea': outputs2}
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
        elif self.cf.task == 'crossmoda':
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                                   sigmoid=False, png=False,
                                   iteration=self.iterations, name='source_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['outputs_source_tumour'], normalize=False, sigmoid=True,
                                   iteration=self.iterations, name='outputs_source_tumour/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['outputs_source_cochlea'], normalize=False, sigmoid=True,
                                   iteration=self.iterations, name='outputs_source_cochlea/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['tumour_labels'], normalize=True, sigmoid=False,
                                   iteration=self.iterations, name='tumour_labels/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['cochlea_labels'], normalize=True, sigmoid=False,
                                   iteration=self.iterations, name='cochlea_labels/{}'.format(split), png=False)
            for key, value in postfix_dict.items():
                self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)

                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.seg_model = self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))