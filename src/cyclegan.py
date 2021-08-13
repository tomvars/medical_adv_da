import itertools
import torch
from .base_model import BaseModel
import torch.optim as optim
import sys
import os
from src.networks import Destilation_student_matchingInstance, SplitHeadModel,\
DiscriminatorCycleGANSimple, GeneratorUnet
from src.utils import to_var_gpu, save_images

class CycleganModel(BaseModel):
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
        self.generator_s_t = GeneratorUnet(1, self.cf.channels, use_sigmoid=True)
        self.generator_t_s = GeneratorUnet(1, self.cf.channels, use_sigmoid=True)

        self.discriminator_s = DiscriminatorCycleGANSimple(1, 2, self.cf.discriminator_complexity)
        self.discriminator_t = DiscriminatorCycleGANSimple(1, 2, self.cf.discriminator_complexity)

        self.optimizer_discriminator = optim.Adam(
            itertools.chain(self.discriminator_s.parameters(), self.discriminator_t.parameters()),
            lr=1e-4)
        self.optimizer_generator = optim.Adam(
            itertools.chain(self.generator_s_t.parameters(), self.generator_t_s.parameters()),
            lr=1e-4)
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
        preds_discriminator_s = self.discriminator_s(torch.cat([generated_source, source_inputs]))
        preds_discriminator_t = self.discriminator_t(torch.cat([target_inputs, generated_target]))
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

            self.discriminator_s.train()
            self.discriminator_t.train()
            self.generator_s_t.train()
            self.generator_t_s.train()
            preds_discriminator_s = self.discriminator_s(torch.cat(
                [generated_source.detach(), source_inputs]))
            preds_discriminator_t = self.discriminator_t(torch.cat(
                [target_inputs, generated_target.detach()]))
            if len(preds_discriminator_s.shape) == 2:
                labels_discriminator_s = to_var_gpu(
                    torch.cat((torch.zeros(generated_source.size(0)),
                               torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
                labels_discriminator_t = to_var_gpu(
                    torch.cat((torch.zeros(generated_target.size(0)),
                               torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
                loss_discriminator_s = self.gan_loss(preds_discriminator_s, labels_discriminator_s)
                loss_discriminator_t = self.gan_loss(preds_discriminator_t, labels_discriminator_t)
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
            if len(preds_discriminator_s.shape) == 2:
                labels_discriminator_s = to_var_gpu(
                    torch.cat((torch.zeros(generated_source.size(0)),
                               torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
                labels_discriminator_t = to_var_gpu(
                    torch.cat((torch.zeros(generated_target.size(0)),
                               torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
                loss_discriminator_s = self.gan_loss(preds_discriminator_s, labels_discriminator_s)
                loss_discriminator_t = self.gan_loss(preds_discriminator_t, labels_discriminator_t)
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
            
        postfix_dict = {'acc_discriminator_s': acc_discriminator_s,
                        'acc_discriminator_t': acc_discriminator_t,
                        'discriminator_t_loss': loss_discriminator_s.item(),
                        'discriminator_s_loss': loss_discriminator_t.item(),
                        'cycle_loss_t': cycle_loss_target.item(),
                        'cycle_loss_s': cycle_loss_source.item()}
        tensorboard_dict = {
            'source_inputs': source_inputs,
            'target_inputs': target_inputs,
            'generated_target': generated_target,
            'generated_source': generated_source,
            'cycle_source': cycle_source,
            'cycle_target': cycle_target
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
                        'cycle_loss_s': cycle_loss_source.item()}
        tensorboard_dict = {
            'source_inputs': source_inputs,
            'target_inputs': target_inputs,
            'generated_target': generated_target,
            'generated_source': generated_source,
            'cycle_source': cycle_source,
            'cycle_target': cycle_target
        }
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split='train'):
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
        for key, value in postfix_dict.items():
            self.writer.add_scalar('{}/{}'.format(key, split), value, self.iterations)
            
    def load(self, checkpoint_path):
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
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
    
    def save(self):
        torch.save({'generator_s_t': self.generator_s_t.state_dict(),
                    'generator_t_s': self.generator_t_s.state_dict(),
                    'discriminator_t': self.discriminator_t.state_dict(),
                    'discriminator_s': self.discriminator_s.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))