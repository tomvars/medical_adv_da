import itertools
import torch

import torch.optim as optim
import sys
import os
from src.base_model import BaseModel
from src.networks import Destilation_student_matchingInstance, SplitHeadModel,\
DiscriminatorCycleGANSimple, GeneratorUnet
from src.pcgrad import PCGrad
from src.utils import to_var_gpu, save_images, dice_soft_loss

class CycadaModel(BaseModel):
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        """
        This is where the models are defined based on the config in cf
        """
        super().__init__()
        self.cf = cf
        self.writer = writer
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.seg_model_source = SplitHeadModel(self.cf.channels)
        self.seg_model_target = SplitHeadModel(self.cf.channels)
        self.generator_s_t = GeneratorUnet(1, self.cf.channels, use_sigmoid=True)
        self.generator_t_s = GeneratorUnet(1, self.cf.channels, use_sigmoid=True)
        self.discriminator_s = DiscriminatorCycleGANSimple(1, 2, self.cf.discriminator_complexity)
        self.discriminator_t = DiscriminatorCycleGANSimple(1, 2, self.cf.discriminator_complexity)
        self.optimizer_seg = PCGrad(optim.Adam(
            itertools.chain(self.seg_model_source.parameters(), self.seg_model_source.parameters()),
            lr=5e-3
        ))
        self.optimizer_discriminator = optim.Adam(
            itertools.chain(self.discriminator_s.parameters(), self.discriminator_t.parameters()),
            lr=self.cf.lr)
        self.optimizer_generator = optim.Adam(
            itertools.chain(self.generator_s_t.parameters(), self.generator_t_s.parameters(),
                           ),
            lr=self.cf.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_generator, milestones=[20000, 20000], gamma=0.1)
        self.criterion_cycle = torch.nn.L1Loss()
        self.gan_loss = torch.nn.CrossEntropyLoss()
        self.spatial_discriminator_loss = torch.nn.MSELoss()
        self.iterations = self.cf.iterations
        
    def initialise(self):
        self.generator_s_t.cuda()
        self.generator_t_s.cuda()
        self.discriminator_s.cuda()
        self.discriminator_t.cuda()
        self.seg_model_source.cuda()
        self.seg_model_target.cuda()
        # Gradient clipping!
        torch.nn.utils.clip_grad_norm(itertools.chain(self.generator_s_t.parameters(),
                                                      self.generator_t_s.parameters(),
                                                      self.discriminator_t.parameters(),
                                                      self.discriminator_s.parameters()), 1.0)

    def training_loop(self, source_dl, target_dl):
        # Training CycleGAN
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        tumour_labels = torch.where(source_labels == 1,
                                    torch.ones_like(source_labels),
                                    torch.zeros_like(source_labels))
        cochlea_labels = torch.where(source_labels == 2,
                                     torch.ones_like(source_labels),
                                     torch.zeros_like(source_labels))
        self.seg_model_source.train()
        self.seg_model_target.train()
        self.discriminator_s.train()
        self.discriminator_t.train()
        self.generator_s_t.train()
        self.generator_t_s.train()
        generated_target = self.generator_s_t(source_inputs)
        generated_source = self.generator_t_s(target_inputs)
        cycle_source = self.generator_t_s(generated_target)
        cycle_target = self.generator_s_t(generated_source)
        cycle_loss_source = self.criterion_cycle(source_inputs, cycle_source)
        cycle_loss_target = self.criterion_cycle(target_inputs, cycle_target)
        cycle_loss = cycle_loss_source + cycle_loss_target

        outputs_source, outputs_source2, _, _, _, _, _, _, _, _, _ = self.seg_model_source(source_inputs)
        outputs_generated_source, outputs_generated_source2, _, _, _, _, _, _, _, _, _  = self.seg_model_source(generated_source.detach())
        outputs_cycle_source, outputs_cycle_source2, _, _, _, _, _, _, _, _, _  = self.seg_model_source(cycle_source.detach())
        
        outputs_target, outputs_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(target_inputs)
        outputs_generated_target, outputs_generated_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(generated_target.detach())
        outputs_cycle_target, outputs_cycle_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(cycle_target.detach())
        
        supervised_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_source), tumour_labels)
        supervised_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_source2), cochlea_labels)
        # Supervised Loss
        supervised_loss = (supervised_tumour_loss + supervised_cochlea_loss) / 2.0
        # Cross-modality semantic consistency
        cmsc_s_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_generated_target), tumour_labels)
        cmsc_s_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_generated_target2), cochlea_labels)
        cmsc_s_loss = (cmsc_s_tumour_loss + cmsc_s_cochlea_loss) / 2.0
        cmsc_t_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_target), torch.sigmoid(outputs_generated_source))
        cmsc_t_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_target2), torch.sigmoid(outputs_generated_source2))
        cmsc_t_loss = (cmsc_t_tumour_loss + cmsc_t_cochlea_loss) / 2.0
        # Intra-modality semantic consistency
        imsc_s_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_source), torch.sigmoid(outputs_cycle_source))
        imsc_s_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_source2), torch.sigmoid(outputs_cycle_source2))
        imsc_s_loss = (imsc_s_tumour_loss + imsc_s_cochlea_loss) / 2.0
        imsc_t_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_target), torch.sigmoid(outputs_cycle_target))
        imsc_t_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_target2), torch.sigmoid(outputs_cycle_target2))
        imsc_t_loss = (imsc_t_tumour_loss + imsc_t_cochlea_loss) / 2.0
        segmentation_losses = [
            cmsc_s_tumour_loss, cmsc_s_cochlea_loss,
            cmsc_t_tumour_loss, cmsc_t_cochlea_loss,
                               imsc_s_tumour_loss, imsc_s_cochlea_loss,
                               imsc_t_tumour_loss, imsc_t_cochlea_loss,
                               supervised_tumour_loss, supervised_cochlea_loss]
        segmentation_loss = 10.0 * cmsc_s_loss + 0.1 * cmsc_t_loss + 0.1 * imsc_s_loss + 0.1 * imsc_t_loss + 10.0 * supervised_loss
        
        preds_discriminator_s = self.discriminator_s(torch.cat([generated_source, source_inputs]))
        preds_discriminator_t = self.discriminator_t(torch.cat([target_inputs, generated_target]))
        preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
        labels_discriminator_s = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
        preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
        labels_discriminator_t = to_var_gpu(
            torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                       torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
        loss_discriminator_s = self.spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
        loss_discriminator_t = self.spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)
        correct_s = (torch.where(preds_discriminator_s > 0.5,
                                 torch.ones_like(preds_discriminator_s),
                                 torch.zeros_like(preds_discriminator_s),
                                ) == labels_discriminator_s).float().sum()
        correct_t = (torch.where(preds_discriminator_t > 0.5,
                                 torch.ones_like(preds_discriminator_t),
                                 torch.zeros_like(preds_discriminator_t),
                                ) == labels_discriminator_t).float().sum()
        num_of_subjects = int(preds_discriminator_s.size(0))
        acc_discriminator_s = (correct_s / num_of_subjects).item()
        acc_discriminator_t = (correct_t / num_of_subjects).item()

        if (acc_discriminator_s > 0.70 and
            acc_discriminator_t > 0.70) and (acc_discriminator_s <= 0.85 and
                                             acc_discriminator_t <= 0.85):
            self.discriminator_s.eval()
            self.discriminator_t.eval()
            self.generator_s_t.train()
            self.generator_t_s.train()
            loss = cycle_loss - loss_discriminator_s - loss_discriminator_t
            self.optimizer_generator.zero_grad()
            loss.backward()
            self.optimizer_generator.step()
            self.optimizer_seg.zero_grad()
            self.optimizer_seg.pc_backward(segmentation_losses)
            self.optimizer_seg.step()
            
#             self.optimizer_seg.zero_grad()
#             segmentation_loss.backward()
#             self.optimizer_seg.step()

            self.discriminator_s.train()
            self.discriminator_t.train()
            self.generator_s_t.train()
            self.generator_t_s.train()
            preds_discriminator_s = self.discriminator_s(torch.cat(
                [generated_source.detach(), source_inputs]))
            preds_discriminator_t = self.discriminator_t(torch.cat(
                [target_inputs, generated_target.detach()]))
            preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_s = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
            preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_t = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
            loss_discriminator_s = self.spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
            loss_discriminator_t = self.spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)

            self.optimizer_discriminator.zero_grad()
            loss_discriminator_s.backward()
            loss_discriminator_t.backward()
            self.optimizer_discriminator.step()
        elif (acc_discriminator_s < 0.70 or acc_discriminator_t < 0.70):
            self.discriminator_s.train()
            self.discriminator_t.train()
            self.generator_s_t.eval()
            self.generator_t_s.eval()
            preds_discriminator_s = self.discriminator_s(torch.cat(
                [generated_source.detach(), source_inputs]))
            preds_discriminator_t = self.discriminator_t(torch.cat(
                [target_inputs, generated_target.detach()]))
            preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_s = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
            preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_t = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
            loss_discriminator_s = self.spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
            loss_discriminator_t = self.spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)
            self.optimizer_discriminator.zero_grad()
            loss_discriminator_s.backward()
            loss_discriminator_t.backward()
            self.optimizer_discriminator.step()
        else:
            self.discriminator_s.eval()
            self.discriminator_t.eval()
            self.generator_s_t.train()
            self.generator_t_s.train()
            loss = cycle_loss - loss_discriminator_s - loss_discriminator_t
            self.optimizer_generator.zero_grad()
            loss.backward()
            self.optimizer_generator.step()
            self.optimizer_seg.zero_grad()
            self.optimizer_seg.pc_backward(segmentation_losses)
            self.optimizer_seg.step()
            
        postfix_dict = {'acc_discriminator_s': acc_discriminator_s,
                        'acc_discriminator_t': acc_discriminator_t,
                        'discriminator_t_loss': loss_discriminator_s.item(),
                        'discriminator_s_loss': loss_discriminator_t.item(),
                        'cycle_loss_t': cycle_loss_target.item(),
                        'cycle_loss_s': cycle_loss_source.item(),
                        'cmsc_s_loss': cmsc_s_loss.item(),
                        'cmsc_s_tumour_loss': cmsc_s_tumour_loss.item(),
                        'cmsc_s_cochlea_loss': cmsc_s_cochlea_loss.item(),
                        'cmsc_t_loss': cmsc_t_loss.item(),
                        'imsc_s_loss': imsc_s_loss.item(),
                        'imsc_t_loss': imsc_t_loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'supervised_tumour_loss': supervised_tumour_loss.item(),
                        'supervised_cochlea_loss': supervised_cochlea_loss.item()
                       }
        tensorboard_dict = {
            'source_inputs': source_inputs,
            'target_inputs': target_inputs,
            'generated_target': generated_target,
            'generated_source': generated_source,
            'cycle_source': cycle_source,
            'cycle_target': cycle_target,
            'tumour_labels': tumour_labels,
            'cochlea_labels': cochlea_labels,
            'outputs_source_tumour': outputs_source,
            'outputs_target_tumour': outputs_target,
            'outputs_source_cochlea': outputs_source2,
            'outputs_target_cochlea': outputs_target2,
            'outputs_generated_target_tumour': outputs_generated_target,
            'outputs_generated_target_cochlea': outputs_generated_target2,
        }
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        # Want output to be 
        # Training CycleGAN
        source_batch = next(source_dl)
        source_inputs, source_labels = (source_batch['inputs'].to(self.device),
                                        source_batch['labels'].to(self.device))
        target_batch = next(target_dl)
        target_inputs, target_labels = (target_batch['inputs'].to(self.device),
                                        target_batch['labels'].to(self.device))
        tumour_labels = torch.where(source_labels == 1,
                                    torch.ones_like(source_labels),
                                    torch.zeros_like(source_labels))
        cochlea_labels = torch.where(source_labels == 2,
                                     torch.ones_like(source_labels),
                                     torch.zeros_like(source_labels))
        self.seg_model_source.eval()
        self.seg_model_target.eval()
        self.discriminator_s.eval()
        self.discriminator_t.eval()
        self.generator_s_t.eval()
        self.generator_t_s.eval()
        generated_target = self.generator_s_t(source_inputs)
        generated_source = self.generator_t_s(target_inputs)
        cycle_source = self.generator_t_s(generated_target)
        cycle_target = self.generator_s_t(generated_source)
        cycle_loss_source = self.criterion_cycle(source_inputs, cycle_source)
        cycle_loss_target = self.criterion_cycle(target_inputs, cycle_target)
        cycle_loss = cycle_loss_source + cycle_loss_target
        preds_discriminator_s = self.discriminator_s(torch.cat([generated_source, source_inputs]))
        preds_discriminator_t = self.discriminator_t(torch.cat([target_inputs, generated_target]))
        
        outputs_source, outputs_source2, _, _, _, _, _, _, _, _, _ = self.seg_model_source(source_inputs)
        outputs_generated_source, outputs_generated_source2, _, _, _, _, _, _, _, _, _  = self.seg_model_source(generated_source)
        outputs_cycle_source, outputs_cycle_source2, _, _, _, _, _, _, _, _, _  = self.seg_model_source(cycle_source)
        outputs_target, outputs_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(target_inputs)
        outputs_generated_target, outputs_generated_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(generated_target)
        outputs_cycle_target, outputs_cycle_target2, _, _, _, _, _, _, _, _, _   = self.seg_model_target(cycle_target)
        supervised_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_source), tumour_labels)
        supervised_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_source2), cochlea_labels)
        # Supervised Loss
        supervised_loss = (supervised_tumour_loss + supervised_cochlea_loss) / 2.0
        # Cross-modality semantic consistency
        cmsc_s_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_generated_target), tumour_labels)
        cmsc_s_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_generated_target2), cochlea_labels)
        cmsc_s_loss = (cmsc_s_tumour_loss + cmsc_s_cochlea_loss) / 2.0
        cmsc_t_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_target), torch.sigmoid(outputs_generated_source))
        cmsc_t_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_target2), torch.sigmoid(outputs_generated_source2))
        cmsc_t_loss = (cmsc_t_tumour_loss + cmsc_t_cochlea_loss) / 2.0
        # Intra-modality semantic consistency
        imsc_s_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_source), torch.sigmoid(outputs_cycle_source))
        imsc_s_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_source2), torch.sigmoid(outputs_cycle_source2))
        imsc_s_loss = (imsc_s_tumour_loss + imsc_s_cochlea_loss) / 2.0
        imsc_t_tumour_loss = dice_soft_loss(torch.sigmoid(outputs_target), torch.sigmoid(outputs_cycle_target))
        imsc_t_cochlea_loss = dice_soft_loss(torch.sigmoid(outputs_target2), torch.sigmoid(outputs_cycle_target2))
        imsc_t_loss = (imsc_t_tumour_loss + imsc_t_cochlea_loss) / 2.0
        segmentation_loss = cmsc_s_loss + cmsc_t_loss + imsc_s_loss + imsc_t_loss + supervised_loss
        
        if len(preds_discriminator_s.shape) == 2:
            labels_discriminator_s = to_var_gpu(
                torch.cat((torch.zeros(generated_source.size(0)),
                           torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
            labels_discriminator_t = to_var_gpu(
                torch.cat((torch.zeros(generated_target.size(0)),
                           torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
            loss_discriminator_s = self.gan_loss(preds_discriminator_s, labels_discriminator_s)
            loss_discriminator_t = self.gan_loss(preds_discriminator_t, labels_discriminator_t)
            correct_s = (torch.argmax(preds_discriminator_s, dim=1) == labels_discriminator_s).float().sum()
            correct_t = (torch.argmax(preds_discriminator_t, dim=1) == labels_discriminator_t).float().sum()
            num_of_subjects = int(preds_discriminator_s.size(0))
            acc_discriminator_s = (correct_s / num_of_subjects).item()
            acc_discriminator_t = (correct_t / num_of_subjects).item()
        else: # Spatial discriminator!
            preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_s = to_var_gpu(
                    torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                               torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
            preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
            labels_discriminator_t = to_var_gpu(
                torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                           torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
            loss_discriminator_s = self.spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
            loss_discriminator_t = self.spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)
            correct_s = (torch.where(preds_discriminator_s > 0.5,
                                     torch.ones_like(preds_discriminator_s),
                                     torch.zeros_like(preds_discriminator_s),
                                    ) == labels_discriminator_s).float().sum()
            correct_t = (torch.where(preds_discriminator_t > 0.5,
                                     torch.ones_like(preds_discriminator_t),
                                     torch.zeros_like(preds_discriminator_t),
                                    ) == labels_discriminator_t).float().sum()
            num_of_subjects = int(preds_discriminator_s.size(0))
            acc_discriminator_s = (correct_s / num_of_subjects).item()
            acc_discriminator_t = (correct_t / num_of_subjects).item()
        postfix_dict = {'acc_discriminator_s': acc_discriminator_s,
                        'acc_discriminator_t': acc_discriminator_t,
                        'discriminator_t_loss': loss_discriminator_s.item(),
                        'discriminator_s_loss': loss_discriminator_t.item(),
                        'cycle_loss_t': cycle_loss_target.item(),
                        'cycle_loss_s': cycle_loss_source.item(),
                        'cmsc_s_loss': cmsc_s_loss.item(),
                        'cmsc_s_tumour_loss': cmsc_s_tumour_loss.item(),
                        'cmsc_s_cochlea_loss': cmsc_s_cochlea_loss.item(),
                        'cmsc_t_loss': cmsc_t_loss.item(),
                        'imsc_s_loss': imsc_s_loss.item(),
                        'imsc_t_loss': imsc_t_loss.item(),
                        'supervised_loss': supervised_loss.item(),
                        'supervised_tumour_loss': supervised_tumour_loss.item(),
                        'supervised_cochlea_loss': supervised_cochlea_loss.item()
                       }
        tensorboard_dict = {
            'source_inputs': source_inputs,
            'target_inputs': target_inputs,
            'generated_target': generated_target,
            'generated_source': generated_source,
            'cycle_source': cycle_source,
            'cycle_target': cycle_target,
            'tumour_labels': tumour_labels,
            'cochlea_labels': cochlea_labels,
            'outputs_source_tumour': outputs_source,
            'outputs_target_tumour': outputs_target,
            'outputs_source_cochlea': outputs_source2,
            'outputs_target_cochlea': outputs_target2,
            'outputs_generated_target_tumour': outputs_generated_target,
            'outputs_generated_target_cochlea': outputs_generated_target2,
        }
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split='train'):
        save_images(writer=self.writer, images=tensorboard_dict['tumour_labels'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='tumour_labels/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['cochlea_labels'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='cochlea_labels/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                                   sigmoid=False, png=False,
                                   iteration=self.iterations, name='source_inputs/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['target_inputs'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='targets_inputs/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['generated_target'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='generated_target/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['generated_source'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='generated_source/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['cycle_source'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='cycle_source/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['cycle_target'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='cycle_target/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_source_tumour'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_source_tumour/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_target_tumour'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_target_tumour/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_source_cochlea'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_source_cochlea/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_target_cochlea'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_target_cochlea/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_generated_target_tumour'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_generated_target_tumour/{}'.format(split))
        save_images(writer=self.writer, images=tensorboard_dict['outputs_generated_target_cochlea'], normalize=True,
                    sigmoid=False, png=False, iteration=self.iterations, name='outputs_generated_target_cochlea/{}'.format(split))
        for key, value in postfix_dict.items():
            self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)
            
    def load(self, checkpoint_path, partial=True):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.generator_s_t.cuda()
        self.generator_t_s.cuda()
        self.discriminator_s.cuda()
        self.discriminator_t.cuda()
        self.generator_s_t.load_state_dict(checkpoint['generator_s_t'])
        self.generator_t_s.load_state_dict(checkpoint['generator_t_s'])
        self.discriminator_t.load_state_dict(checkpoint['discriminator_t'])
        self.discriminator_s.load_state_dict(checkpoint['discriminator_s'])
#         self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
#         if 'optimizer_seg' in checkpoint:
#             self.optimizer_seg.load_state_dict(checkpoint['optimizer_seg'])
        if 'seg_model_source' in checkpoint:
            self.seg_model_source.cuda()
            self.seg_model_source.load_state_dict(checkpoint['seg_model_source'])
        if 'seg_model_target' in checkpoint:
            self.seg_model_target.cuda()
            self.seg_model_target.load_state_dict(checkpoint['seg_model_target'])
            
    def save(self):
        torch.save({'generator_s_t': self.generator_s_t.state_dict(),
                    'generator_t_s': self.generator_t_s.state_dict(),
                    'discriminator_t': self.discriminator_t.state_dict(),
                    'discriminator_s': self.discriminator_s.state_dict(),
                    'seg_model_source': self.seg_model_source.state_dict(),
                    'seg_model_target': self.seg_model_target.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
#                     'optimizer_seg': self.optimizer_seg.state_dict()
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))