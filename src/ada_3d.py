import multiprocessing
import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from functools import partial
from monai.losses.dice import DiceLoss, DiceCELoss
from monai.networks.nets import BasicUNet
from src.base_model import BaseModel
from src.networks import Discriminator3D, BasicUNetFeatures
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss,\
ss_loss, generate_affine, non_geometric_augmentations, apply_transform, create_overlap_image

class ADA3DModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.seg_model = BasicUNetFeatures(spatial_dims=3,
                                           in_channels=self.cf.channels,
                                           out_channels=2)
        self.seg_model.cuda()
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.cf.lr)
        step_1 = 20000 if self.cf.data_task == 'ms' else 5000
        step_2 = 20000 if self.cf.data_task == 'ms' else 10000
        scheduler_S = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        self.self_supervised_loss = dice_soft_loss
        self.supervised_loss = DiceLoss(softmax=True)
        self.dice_ce_loss = DiceCELoss(softmax=True, to_onehot_y=True)
        self.iterations = self.cf.iterations
        # Discriminator setup #
        self.discriminator = Discriminator3D(256, 2, self.cf.discriminator_complexity)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        ########################
        self.correct = 0
        self.num_of_subjects = 0

    def initialise(self):
        self.seg_model.cuda()
        self.discriminator.cuda()
        
    def training_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights
        source_batch = next(source_dl)
        source_inputs, source_labels = (torch.tensor(np.stack([f[0]['inputs'] for f in source_batch])).to(self.device),
                                        torch.tensor(np.stack([f[0]['labels'] for f in source_batch])).to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        source_outputs, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)
        # Training Discriminator
        self.seg_model.eval()
        self.discriminator.train()
        inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
        labels_discriminator = torch.cat((torch.zeros(source_inputs.size(0)),
                                          torch.ones(target_inputs_aug.size(0))),
                                         0).type(torch.LongTensor).to(self.device)
        _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
        dec1 = F.interpolate(dec1, size=dec1.size()[2:], mode='trilinear')
        dec2 = F.interpolate(dec2, size=dec1.size()[2:], mode='trilinear')
        dec3 = F.interpolate(dec3, size=dec1.size()[2:], mode='trilinear')
        dec4 = F.interpolate(dec4, size=dec1.size()[2:], mode='trilinear')
        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)

        self.discriminator.zero_grad()
        outputs_discriminator = self.discriminator(inputs_discriminator)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        loss_discriminator.backward()
        self.discriminator_optimizer.step()
        self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
        self.num_of_subjects += int(outputs_discriminator.size(0))

        
        # Train seg model
        self.seg_model.train()
        self.discriminator.eval()
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        target_outputs, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs)
        target_outputs_aug, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs_aug)
        affines = torch.tensor(np.stack([f[0]['inputs_aug_transforms'][-1]['extra_info']['affine'] for f in target_batch])).to(self.device)
        grid = F.affine_grid(affines[:, :3, :],
                             size=[self.cf.batch_size] + [1] + self.cf.spatial_size
                            ).type(torch.FloatTensor).to(self.device)
        target_outputs_transformed = F.grid_sample(target_outputs,
                                                   grid=grid, padding_mode="border")
        pc_loss = alpha * self.self_supervised_loss(torch.sigmoid(target_outputs_aug[:, 1:, ...]),
                                                    torch.sigmoid(target_outputs_transformed[:, 1:, ...]))
        inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
        _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
        dec1 = F.interpolate(dec1, size=dec1.size()[2:], mode='trilinear')
        dec2 = F.interpolate(dec2, size=dec1.size()[2:], mode='trilinear')
        dec3 = F.interpolate(dec3, size=dec1.size()[2:], mode='trilinear')
        dec4 = F.interpolate(dec4, size=dec1.size()[2:], mode='trilinear')
        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)
        outputs_discriminator = self.discriminator(inputs_discriminator)
        labels_discriminator = torch.cat((torch.zeros(source_inputs.size(0)),
                                          torch.ones(target_inputs_aug.size(0))),
                                         0).type(torch.LongTensor).to(self.device)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
#         supervised_loss = self.supervised_loss(source_outputs, source_labels)
#         supervised_loss = dice_soft_loss(torch.sigmoid(source_outputs[:, 1:, ...]), source_labels)
        supervised_loss = self.dice_ce_loss(source_outputs, source_labels)
        adversarial_loss = - beta * loss_discriminator
        self.seg_optimizer.zero_grad()
        loss = supervised_loss + pc_loss + adversarial_loss
        loss.backward()
        self.seg_optimizer.step()
        
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'pc_loss': pc_loss.item(),
                        'adversarial_loss': adversarial_loss.item(),
                        'loss_discriminator': loss_discriminator.item(),
                        'acc_discriminator': (self.correct/self.num_of_subjects).item()
                       }
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'target_inputs_aug': target_inputs_aug,
                            'source_labels': source_labels,
                            'source_outputs': source_outputs[:, 1:, ...],
                            'target_outputs': target_outputs[:, 1:, ...]}
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights 
        source_batch = next(source_dl)
        source_inputs, source_labels = (torch.tensor(np.stack([f[0]['inputs'] for f in source_batch])).to(self.device),
                                        torch.tensor(np.stack([f[0]['labels'] for f in source_batch])).to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        source_outputs, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)

        # Eval seg model
        self.seg_model.eval()
        self.discriminator.eval()
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        target_outputs, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs)
        target_outputs_aug, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs_aug)
        affines = torch.tensor(np.stack([f[0]['inputs_aug_transforms'][-1]['extra_info']['affine'] for f in target_batch])).to(self.device)
        grid = F.affine_grid(affines[:, :3, :],
                             size=[self.cf.batch_size] + [1] + self.cf.spatial_size
                            ).type(torch.FloatTensor).to(self.device)
        target_outputs_transformed = F.grid_sample(target_outputs,
                                                   grid=grid, padding_mode="border")
        pc_loss = alpha * self.self_supervised_loss(torch.sigmoid(target_outputs_aug[:, 1:, ...]),
                                                    torch.sigmoid(target_outputs_transformed[:, 1:, ...]))
        inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
        _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
        dec1 = F.interpolate(dec1, size=dec1.size()[2:], mode='trilinear')
        dec2 = F.interpolate(dec2, size=dec1.size()[2:], mode='trilinear')
        dec3 = F.interpolate(dec3, size=dec1.size()[2:], mode='trilinear')
        dec4 = F.interpolate(dec4, size=dec1.size()[2:], mode='trilinear')
        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)
        outputs_discriminator = self.discriminator(inputs_discriminator)
        labels_discriminator = torch.cat((torch.zeros(source_inputs.size(0)),
                                          torch.ones(target_inputs_aug.size(0))),
                                         0).type(torch.LongTensor).to(self.device)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
        self.num_of_subjects += int(outputs_discriminator.size(0))
        supervised_loss = dice_soft_loss(torch.sigmoid(source_outputs[:, 1:, ...]), source_labels)
        adversarial_loss = - beta * loss_discriminator        
        loss = supervised_loss + pc_loss + adversarial_loss
        
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'pc_loss': pc_loss.item(),
                        'adversarial_loss': adversarial_loss.item(),
                        'loss_discriminator': loss_discriminator.item(),
                        'acc_discriminator': (self.correct/self.num_of_subjects).item()
                       }
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'target_inputs_aug': target_inputs_aug,
                            'source_labels': source_labels,
                            'source_outputs': source_outputs[:, 1:, ...],
                            'target_outputs': target_outputs[:, 1:, ...]}
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.data_task == 'tumour':
            for idx, modality in enumerate(['flair', 't1c', 't1', 't2']):
                save_images(writer=self.writer, images=tensorboard_dict['source_inputs'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name='source_{}/{}'.format(modality, split))
                save_images(writer=self.writer, images=tensorboard_dict['target_inputs'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name='target_{}/{}'.format(modality, split))
                save_images(writer=self.writer, images=tensorboard_dict['inputstaug'][:, (idx,), :, :],
                            normalize=True, sigmoid=False,
                            iteration=self.iterations, name='{}_aug/{}'.format(modality, split))
        elif self.cf.data_task in ['ms_3d', 'microbleed_3d', 'crossmoda_3d']:
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='source_labels/{}'.format(split), png=False)
#             save_images(writer=self.writer, images=tensorboard_dict['target_labels'], normalize=True, sigmoid=False,
#                         iteration=self.iterations, name='target_labels/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['source_outputs'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='source_outputs/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                        sigmoid=False, png=False,
                        iteration=self.iterations, name='source_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_inputs'], normalize=True,
                        sigmoid=False, png=False,
                        iteration=self.iterations, name='targets_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_inputs_aug'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='target_inputs_aug/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_outputs'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='target_outputs/{}'.format(split))
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['source_inputs'],
                                 preds=tensorboard_dict['source_outputs'],
                                 labels=tensorboard_dict['source_labels'],
                                 sigmoid=True,
                                 iteration=self.iterations, name='source_confusion_plot/{}'.format(split))
            
        for key, value in postfix_dict.items():
            self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)
                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.seg_model = self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.discriminator = self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_discriminator = self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_discriminator'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'optimizer_discriminator': self.discriminator_optimizer.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
        
    def epoch_reset(self):
        self.correct = 0
        self.num_of_subjects = 0