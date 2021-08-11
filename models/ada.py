import multiprocessing
import sys
sys.path.append('../')
import torch.optim as optim
import numpy as np
from functools import partial
from .base_model import BaseModel
from networks import Destilation_student_matchingInstance
from utils import save_images
from utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform

class ADAModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
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
        scheduler_S = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
        # Discriminator setup #
        self.discriminator = DiscriminatorDomain(352, 2, self.cf.discriminator_complexity)
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        ########################
        self.correct = 0
        self.num_of_subjects = 0

    def initialise(self):
        self.seg_model.cuda()
        self.discriminator.cuda()
        self.p = multiprocessing.Pool(10)
        
    def training_loop(self, source_dl, target_dl):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights 
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        outputs, _, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)
        # Training Discriminator
        self.seg_model.eval()
        self.discriminator.train()

        # do the sampling here.

        # Source Domain sampling
        inputs_source_discriminator = source_inputs
        # Target batch
        batch_trs = target_inputs.cpu().numpy()
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_trs))
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_trs))
        inputs_target_discriminator_aug = torch.Tensor(batch_trs).cuda()

        Theta, Theta_inv = generate_affine(inputs_target_discriminator_aug,
                                          degreeFreedom=self.cf.affine_rot_degree, scale=self.cf.affine_scale,
                                          shearingScale=self.cf.affine_shearing)
        inputs_target_discriminator_aug = apply_transform(inputs_target_discriminator_aug, Theta)

        inputs_models_discriminator = torch.cat(
            (inputs_source_discriminator, inputs_target_discriminator_aug), 0)
        labels_discriminator = to_var_gpu(
            torch.cat((torch.zeros(inputs_source_discriminator.size(0)),
                       torch.ones(inputs_target_discriminator_aug.size(0))), 0).type(torch.LongTensor))

        # print('size Discriminator')
        # print(inputs_models_discriminator.size())

        _, _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)

        dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
        dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
        dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
        dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')

        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)

        self.discriminator.zero_grad()
        outputs_discriminator = self.discriminator(inputs_discriminator)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
        self.num_of_subjects += int(outputs_discriminator.size(0))

        loss_discriminator.backward()
        self.optimizer_discriminator.step()
        discriminator_loss = loss_discriminator.item()
        # Train model
        self.seg_model.train()
        self.discriminator.eval()

        # Here we get a new batch of target domain slices
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(device),
                                        target_batch['labels'].to(device))
        outputst, _, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs)
        batch_trs = target_inputs.cpu().numpy()

        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_trs))
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_trs))

        inputstaug = torch.Tensor(batch_trs).cuda()
        Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=self.cf.affine_rot_degree,
                                          scale=self.cf.affine_scale,
                                          shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, Theta)

        self.seg_model.zero_grad()
        outputstaug, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        outputst_transformed = apply_transform(outputst, Theta)

        inputs_models_discriminator = torch.cat((source_inputs, inputstaug), 0)
        _, _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)

        dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
        dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
        dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
        dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')

        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)

        outputs_discriminator = self.discriminator(inputs_discriminator)
        labels_discriminator = to_var_gpu(
            torch.cat((torch.zeros(source_inputs.size(0)),
                       torch.ones(inputstaug.size(0))), 0).type(torch.LongTensor))
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
        pc_loss = alpha * self.criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
        adversarial_loss = - beta * loss_discriminator
        loss = supervised_loss + pc_loss + adversarial_loss
        self.seg_model.zero_grad()
        loss.backward()
        self.seg_optimizer.step()
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'pc_loss': pc_loss.item(),
                        'adversarial_loss': adversarial_loss.item(),
                        'loss_discriminator': loss_discriminator.item(),
                        'acc_discriminator': self.correct/self.num_of_subjects
                       }
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'source_labels': source_labels,
                            'target_labels': target_labels,
                            'inputstaug': inputstaug,
                            'outputs': outputs,
                            'outputst': outputst}
    def validation_loop(self):
        if self.iterations < self.cf.iterations_adapt:
            alpha = 0
            beta = 0
        else:
            alpha = self.cf.alpha_lweights
            beta = self.cf.beta_lweights 
        self.seg_model.eval()
        self.discriminator.eval()
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        outputs, _, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)
        # Source Domain sampling
        inputs_source_discriminator = source_inputs
        # Target batch
        batch_trs = target_inputs.cpu().numpy()
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_trs))
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_trs))
        inputs_target_discriminator_aug = torch.Tensor(batch_trs).cuda()
        Theta, Theta_inv = generate_affine(inputs_target_discriminator_aug,
                                          degreeFreedom=self.cf.affine_rot_degree, scale=self.cf.affine_scale,
                                          shearingScale=self.cf.affine_shearing)
        inputs_target_discriminator_aug = apply_transform(inputs_target_discriminator_aug, Theta)
        inputs_models_discriminator = torch.cat(
            (inputs_source_discriminator, inputs_target_discriminator_aug), 0)
        labels_discriminator = to_var_gpu(
            torch.cat((torch.zeros(inputs_source_discriminator.size(0)),
                       torch.ones(inputs_target_discriminator_aug.size(0))), 0).type(torch.LongTensor))
        _, _, _, _, _, _, dec4, dec3, dec2, dec1 = self.seg_model(inputs_models_discriminator)
        dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
        dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
        dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
        dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')
        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)
        outputs_discriminator = self.discriminator(inputs_discriminator)
        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                          labels_discriminator)
        self.correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
        self.num_of_subjects += int(outputs_discriminator.size(0))
        
        # Here we get a new batch of target domain slices
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(device),
                                        target_batch['labels'].to(device))
        outputst, _, _, _, _, _, _, _, _, _ = self.seg_model(target_inputs)
        batch_trs = target_inputs.cpu().numpy()

        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_trs))
        batch_trs = self.p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_trs))

        inputstaug = torch.Tensor(batch_trs).cuda()
        Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=self.cf.affine_rot_degree,
                                          scale=self.cf.affine_scale,
                                          shearingScale=self.cf.affine_shearing)
        inputstaug = apply_transform(inputstaug, Theta)

        outputstaug, _, _, _, _, _, _, _, _, _ = self.seg_model(inputstaug)
        outputst_transformed = apply_transform(outputst, Theta)
        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
        pc_loss = alpha * self.criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
        adversarial_loss = - beta * loss_discriminator
        loss = supervised_loss + pc_loss + adversarial_loss
        
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'pc_loss': pc_loss.item(),
                        'adversarial_loss': adversarial_loss.item(),
                        'loss_discriminator': loss_discriminator.item(),
                        'acc_discriminator': self.correct/self.num_of_subjects
                       }
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'source_labels': source_labels,
                            'target_labels': target_labels,
                            'inputstaug': inputstaug,
                            'outputs': outputs,
                            'outputst': outputst}
    def tensorboard_logging(self, tensorboard_dict, split):
        if self.cf.task == 'tumour':
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
        elif self.cf.task == 'ms':
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='source_labels/{}'.format(split), png=True)
            save_images(writer=self.writer, images=tensorboard_dict['target_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='target_labels/{}'.format(split), png=True)
            save_images(writer=self.writer, images=tensorboard_dict['outputs'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='outputs_source/{}'.format(split), png=True)
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                        sigmoid=False, png=True,
                        iteration=self.iterations, name='source_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['target_inputs'], normalize=True,
                        sigmoid=False, png=True,
                        iteration=self.iterations, name='targets_inputs/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['inputstaug'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='inputsaug/{}'.format(split))
            save_images(writer=self.writer, images=tensorboard_dict['outputst'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='outputs_target/{}'.format(split))
                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.seg_model = self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.discriminator = self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_discriminator = self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
        
    def epoch_reset(self):
        self.correct = 0
        self.num_of_subjects = 0