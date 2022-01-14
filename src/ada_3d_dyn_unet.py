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
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.losses.spatial_mask import MaskedLoss
from monai.transforms import Affine
from src.custom_monai_transforms import DiceCESurfaceLoss, SurfaceLoss
from src.base_model import BaseModel
from src.networks import Discriminator3D_retina, BasicUNetFeatures, bigUNet
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss,\
ss_loss, generate_affine, non_geometric_augmentations, apply_transform, create_overlap_image

class ADA3DDynUNETModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        super().__init__()
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.seg_model = bigUNet(spatial_dims=3,
                                 in_channels=self.cf.channels,
                                 out_channels=2,
                                 features=(64, 96, 128, 192, 256, 384, 512, 64) #(32, 48, 64, 96, 128, 192, 256, 32)
                                )
        self.seg_model.cuda()
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.cf.lr)
#         step_1 = 20000 if self.cf.data_task == 'ms' else 5000
#         step_2 = 20000 if self.cf.data_task == 'ms' else 10000
#         scheduler_S = optim.lr_scheduler.CosineAnnealingLR(self.seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        self.seg_scheduler = WarmupCosineSchedule(
            optimizer=self.seg_optimizer,
            warmup_steps=250,
            t_total=self.cf.iterations
        )
        self.self_supervised_loss = dice_soft_loss
        self.masked_ssl_loss = MaskedLoss(dice_soft_loss)
        self.dice_ce_loss = DiceCELoss(softmax=True, to_onehot_y=True)
        self.surface_loss = SurfaceLoss(indices=(1,))
        self.iterations = self.cf.iterations
        # Discriminator setup #
        self.discriminator = Discriminator3D_retina(1120, 2, self.cf.discriminator_complexity).cuda()
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        ########################
        self.correct = 0
        self.num_of_subjects = 0
        self.discriminator_acc = 0.5

    def initialise(self):
        pass
        
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
        if source_inputs.shape != (self.cf.batch_size,
                                   1, self.cf.spatial_size[0], self.cf.spatial_size[1], self.cf.spatial_size[2]):
            print(f'bad shape {source_inputs.shape}')
            return None, None
        if self.cf.use_boundary_loss:
            source_distmaps = torch.tensor(np.stack([f[0]['labels_distmap'] for f in source_batch])).to(self.device)
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        source_outputs, _, _, _, _, _, _ = self.seg_model(source_inputs)
        if self.discriminator_acc < 0.80:
            # Training Discriminator
            self.seg_model.eval()
            self.discriminator.train()
            inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
            
            _, dec6, dec5, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
            dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear')
            dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear')
            dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear')
            dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear')
            dec5 = F.interpolate(dec5, size=dec3.size()[2:], mode='trilinear')
            dec6 = F.interpolate(dec6, size=dec3.size()[2:], mode='trilinear')
            
            inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4, dec5, dec6), 1)
            
            self.discriminator.zero_grad()
            outputs_discriminator = self.discriminator(inputs_discriminator)
            labels_discriminator = torch.cat((torch.zeros_like(dec3)[:source_inputs.size(0), 0, ...],
                                              torch.ones_like(dec3)[:target_inputs_aug.size(0), 0, ...]),
                                             0).type(torch.LongTensor).to(self.device)
            loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                              labels_discriminator)
            loss_discriminator.backward()
            self.discriminator_optimizer.step()
            del dec1, dec2, dec3, dec4, dec5, dec6
            torch.cuda.empty_cache()
        
        
        
        # Train seg model
        self.seg_model.train()
        self.discriminator.eval()
        target_batch = next(target_dl)
        target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                            torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
        with torch.no_grad():
            target_outputs, _, _, _, _, _, _ = self.seg_model(target_inputs)
        target_outputs_aug, _, _, _, _, _, _ = self.seg_model(target_inputs_aug)
#         import pdb; pdb.set_trace()
        affines = torch.tensor(np.stack([f[0]['inputs_aug_transforms'][-2]['extra_info']['affine'] for f in target_batch])).to(self.device)
        target_outputs_transformed = []
        for idx in range(target_outputs.shape[0]):
            affine_op = Affine(affine=affines[idx, ...].cpu().numpy(), image_only=True, padding_mode="zeros")
            target_outputs_transformed.append(affine_op(target_outputs[idx, ...]))
        target_outputs_transformed = torch.stack(target_outputs_transformed)
        # pseudo-code from conversation with Jorge
        mask = torch.ones_like(target_outputs_aug[:, 0:1, ...])
        target_outputs_transformed_mask = []
        for idx in range(target_outputs.shape[0]):
            affine_op = Affine(affine=affines[idx, ...], image_only=True, padding_mode="zeros")
            target_outputs_transformed_mask.append(affine_op(mask[idx, ...]))
        target_outputs_transformed_mask = (torch.stack(target_outputs_transformed_mask) > 0.9).long()
        #######
#         pc_loss = alpha * self.self_supervised_loss(torch.sigmoid(target_outputs_aug[:, 1:, ...]),
#                                                      torch.sigmoid(target_outputs_transformed[:, 1:, ...]))
        pc_loss = alpha * self.masked_ssl_loss(input=torch.sigmoid(target_outputs_aug[:, 1:, ...]),
                                               target=torch.sigmoid(target_outputs_transformed[:, 1:, ...]),
                                               mask=target_outputs_transformed_mask)
                           #* target_outputs_transformed_mask)
        inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
        _, dec6, dec5, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
        dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='trilinear')
        dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='trilinear')
        dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear')
        dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear')
        dec5 = F.interpolate(dec5, size=dec3.size()[2:], mode='trilinear')
        dec6 = F.interpolate(dec6, size=dec3.size()[2:], mode='trilinear')
        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4, dec5, dec6), 1)
        outputs_discriminator = self.discriminator(inputs_discriminator)
        labels_discriminator = torch.cat((torch.zeros_like(dec3)[:source_inputs.size(0), 0, ...],
                                              torch.ones_like(dec3)[:target_inputs_aug.size(0), 0, ...]),
                                             0).type(torch.LongTensor).to(self.device)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
        self.num_of_subjects += int(outputs_discriminator.numel() / 2)
        self.discriminator_acc = (self.correct/self.num_of_subjects).item()
        del dec1, dec2, dec3, dec4, dec5, dec6

#         supervised_loss = self.supervised_loss(source_outputs, source_labels)
#         supervised_loss = dice_soft_loss(torch.sigmoid(source_outputs[:, 1:, ...]), source_labels)
        if self.discriminator_acc > 0.7 and self.iterations > self.cf.iterations_adapt:
            beta = self.cf.beta_lweights
        else:
            beta = 0.0
        supervised_loss = 0
        supervised_dice_ce_loss = self.dice_ce_loss(source_outputs, source_labels)
        if self.cf.use_boundary_loss:
            supervised_surface_loss = self.surface_loss(probs=source_outputs, distmaps=source_distmaps)
            supervised_loss = supervised_surface_loss + supervised_dice_ce_loss
        else:
            supervised_loss = supervised_dice_ce_loss
        adversarial_loss = - beta * loss_discriminator
        self.seg_optimizer.zero_grad()
        loss = supervised_loss + alpha *  pc_loss + adversarial_loss
        loss.backward()
        self.seg_optimizer.step()
        self.seg_scheduler.step()
        # For debugging
        with torch.no_grad():
            target_labels = torch.tensor(np.stack([f[0]['labels'] for f in target_batch])).to(self.device)
            target_loss = self.self_supervised_loss(target_outputs, target_labels)
        
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'pc_loss': pc_loss.item(),
                        'adversarial_loss': adversarial_loss.item(),
                        'loss_discriminator': loss_discriminator.item(),
                        'acc_discriminator': (self.correct/self.num_of_subjects).item(),
                        'target_loss': target_loss.item()
                       }
        if self.cf.use_boundary_loss:
            postfix_dict['surf_loss'] = supervised_surface_loss.item()
        tensorboard_dict = {'source_inputs': source_inputs.detach(),
                            'target_inputs': target_inputs.detach(),
                            'target_inputs_aug': target_inputs_aug.detach(),
                            'target_labels': target_labels.detach(),
                            'source_labels': source_labels.detach(),
                            'source_outputs': source_outputs[:, 1:, ...].detach(),
                            'target_outputs_transformed_mask': target_outputs_transformed_mask.detach(),
                            'target_outputs': target_outputs[:, 1:, ...].detach(),
                            'target_outputs_transformed': target_outputs_transformed[:, 1:, ...].detach(),
                            'target_outputs_aug': target_outputs_aug[:, 1:, ...].detach()
                           }
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights
        with torch.no_grad():
            # Eval seg model
            self.seg_model.eval()
            self.discriminator.eval()
            source_batch = next(source_dl)
            source_inputs, source_labels = (torch.tensor(np.stack([f[0]['inputs'] for f in source_batch])).to(self.device),
                                           torch.tensor(np.stack([f[0]['labels'] for f in source_batch])).to(self.device))
            if source_inputs.shape != (self.cf.batch_size,
                                       1, self.cf.spatial_size[0], self.cf.spatial_size[1], self.cf.spatial_size[2]):
                print(f'bad shape {source_inputs.shape}')
                return None, None
            if self.cf.use_boundary_loss:
                source_distmaps = torch.tensor(np.stack([f[0]['labels_distmap'] for f in source_batch])).to(self.device)
            target_batch = next(target_dl)
            target_inputs, target_inputs_aug = (torch.tensor(np.stack([f[0]['inputs'] for f in target_batch])).to(self.device),
                                                torch.tensor(np.stack([f[0]['inputs_aug'] for f in target_batch])).to(self.device))
            source_outputs, _, _, _, _, _, _ = self.seg_model(source_inputs)
            
            # For debugging
            target_labels = torch.tensor(np.stack([f[0]['labels'] for f in target_batch])).to(self.device)
            
            target_outputs, _, _, _, _, _, _ = self.seg_model(target_inputs)
            target_outputs_aug, _, _, _, _, _, _ = self.seg_model(target_inputs_aug)
            affines = torch.tensor(np.stack([f[0]['inputs_aug_transforms'][-2]['extra_info']['affine'] for f in target_batch])).to(self.device)
            target_outputs_transformed = []
            for idx in range(target_outputs.shape[0]):
                affine_op = Affine(affine=affines[0, ...], image_only=True, padding_mode="zeros")
                target_outputs_transformed.append(affine_op(target_outputs[0, ...]))
            target_outputs_transformed = torch.stack(target_outputs_transformed)
            # pseudo-code from conversation with Jorge
            mask = torch.ones_like(target_outputs_aug[:, 0:1, ...])
            target_outputs_transformed_mask = []
            for idx in range(target_outputs.shape[0]):
                affine_op = Affine(affine=affines[0, ...], image_only=True, padding_mode="zeros")
                target_outputs_transformed_mask.append(affine_op(mask[0, ...]))
            target_outputs_transformed_mask = torch.stack(target_outputs_transformed_mask)
            pc_loss = alpha * self.masked_ssl_loss(input=torch.sigmoid(target_outputs_aug[:, 1:, ...]),
                                               target=torch.sigmoid(target_outputs_transformed[:, 1:, ...]),
                                               mask=target_outputs_transformed_mask)
            inputs_models_discriminator = torch.cat((source_inputs, target_inputs_aug), 0)
            _, dec6, dec5, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
            dec1 = F.interpolate(dec1, size=dec3.size()[2:], mode='area')
            dec2 = F.interpolate(dec2, size=dec3.size()[2:], mode='area')
            dec3 = F.interpolate(dec3, size=dec3.size()[2:], mode='trilinear')
            dec4 = F.interpolate(dec4, size=dec3.size()[2:], mode='trilinear')
            dec5 = F.interpolate(dec5, size=dec3.size()[2:], mode='trilinear')
            dec6 = F.interpolate(dec6, size=dec3.size()[2:], mode='trilinear')
            inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4, dec5, dec6), 1)
            outputs_discriminator = self.discriminator(inputs_discriminator)
            labels_discriminator = torch.cat((torch.zeros_like(dec3)[:source_inputs.size(0), 0, ...],
                                              torch.ones_like(dec3)[:target_inputs_aug.size(0), 0, ...]),
                                             0).type(torch.LongTensor).to(self.device)
            loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                              labels_discriminator)
            self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
            self.num_of_subjects += int(outputs_discriminator.numel() / 2)
            supervised_loss = 0
            supervised_dice_ce_loss = self.dice_ce_loss(source_outputs, source_labels)
            if self.cf.use_boundary_loss:
                supervised_surface_loss = self.surface_loss(probs=source_outputs, distmaps=source_distmaps)
                supervised_loss = supervised_surface_loss + supervised_dice_ce_loss
            else:
                supervised_loss = supervised_dice_ce_loss
            adversarial_loss = - beta * loss_discriminator        
            loss = supervised_loss + pc_loss + adversarial_loss
            # For debugging
            target_labels = torch.tensor(np.stack([f[0]['labels'] for f in target_batch])).to(self.device)
            target_loss = self.self_supervised_loss(target_outputs, target_labels)

            postfix_dict = {
                'loss': loss.item(),
                'supervised_loss': supervised_loss.item(),
                'pc_loss': pc_loss.item(),
                'adversarial_loss': adversarial_loss.item(),
                'loss_discriminator': loss_discriminator.item(),
                'acc_discriminator': (self.correct/self.num_of_subjects).item(),
                'target_loss': target_loss.item()
               }
            if self.cf.use_boundary_loss:
                postfix_dict['surf_loss'] = supervised_surface_loss.item()
            tensorboard_dict = {'source_inputs': source_inputs.detach(),
                                'target_inputs': target_inputs.detach(),
                                'target_inputs_aug': target_inputs_aug.detach(),
                                'target_labels': target_labels.detach(),
                                'source_labels': source_labels.detach(),
                                'source_outputs': source_outputs[:, 1:, ...].detach(),
                                'target_outputs_transformed_mask': target_outputs_transformed_mask.detach(),
                                'target_outputs': target_outputs[:, 1:, ...].detach(),
                                'target_outputs_transformed': target_outputs_transformed[:, 1:, ...].detach(),
                                'target_outputs_aug': target_outputs_aug[:, 1:, ...].detach()
                               }
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.data_task in ['ms', 'microbleed', 'crossmoda', 'tumour']:
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='source_labels/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['target_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='target_labels/{}'.format(split), png=False)
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
            save_images(writer=self.writer, images=tensorboard_dict['target_outputs_transformed'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='target_outputs_transformed/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_outputs_transformed_mask'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='target_outputs_transformed_mask/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_outputs_aug'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='target_outputs_aug/{}'.format(split))
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['source_inputs'],
                                 preds=tensorboard_dict['source_outputs'],
                                 labels=tensorboard_dict['source_labels'],
                                 sigmoid=True,
                                 iteration=self.iterations, name='source_confusion_plot/{}'.format(split))
            create_overlap_image(writer=self.writer,
                                 images=tensorboard_dict['target_inputs'],
                                 preds=tensorboard_dict['target_outputs'],
                                 labels=tensorboard_dict['target_labels'],
                                 sigmoid=True,
                                 iteration=self.iterations, name='target_confusion_plot/{}'.format(split))
            
        for key, value in postfix_dict.items():
            self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)
                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_discriminator'])
        self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'optimizer_discriminator': self.discriminator_optimizer.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
        
    def epoch_reset(self):
        self.correct = 0
        self.num_of_subjects = 0
        
    def inference_func(self, inference_inputs):
        with torch.no_grad():
            inference_outputs, _, _, _, _, _, _  = self.seg_model(inference_inputs)
        return (torch.sigmoid(inference_outputs[:, 1:, ...]) > 0.5).float()
