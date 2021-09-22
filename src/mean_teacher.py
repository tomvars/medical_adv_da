import multiprocessing
import sys
import torch.optim as optim
import numpy as np
from functools import partial
from src.networks import Destilation_student_matchingInstance
from src.utils import save_images
from src.utils import bland_altman_loss, dice_soft_loss, ss_loss, generate_affine, non_geometric_augmentations, apply_transform, update_ema_variables

class MeanTeacherModel:
    def __init__(self, cf, writer, results_folder, models_folder, tensorboard_folder,
                 run_name, starting_epoch=0):
        self.cf = cf
        self.results_folder = results_folder
        self.models_folder = models_folder
        self.tensorboard_folder = tensorboard_folder
        self.run_name = run_name
        self.starting_epoch = starting_epoch
        self.student_model = Destilation_student_matchingInstance(self.cf.labels - 1, self.cf.channels)
        self.writer = writer
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.cf.lr)
        step_1 = 20000 if self.cf.data_task == 'ms' else 5000
        step_2 = 20000 if self.cf.data_task == 'ms' else 10000
        scheduler_S = optim.lr_scheduler.MultiStepLR(self.seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        self.criterion = dice_soft_loss if self.cf.loss == 'dice' else bland_altman_loss
        self.criterion2 = ss_loss
        self.iterations = self.cf.iterations
        self.teacher_model = Destilation_student_matchingInstance(self.cf.labels - 1, self.cf.channels)
            
    def initialise(self):
        for param in self.teacher_model.parameters():
            param.detach_()
        self.studen_model.cuda()
        self.teacher_model.cuda()
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
        self.student_model.train()
        self.teacher_model.train()
        batch_trs = target_inputs.cpu().numpy()
        batch_trs = p.map(
            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
            np.copy(batch_trs))
        batch_trs = p.map(
            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
            np.copy(batch_trs))
        inputstaug = torch.Tensor(batch_trs).cuda()
        Theta, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                          scale=args.affine_scale,
                                          shearingScale=args.affine_shearing)
        inputstaug = apply_transform(inputstaug, Theta)
        self.student_model.train()
        self.teacher_model.train()
        outputs, _, _, _, _, _, _, _, _, _ = self.student_model(source_inputs)
        outputstaug, _, _, _, _, _, _, _, _, _ = self.student_model(inputstaug)
        with torch.no_grad():
            outputst, _, _, _, _, _, _, _, _, _ = self.teacher_model(target_inputs)
        outputst_transformed = apply_transform(outputst, Theta)
        pc_loss = self.criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
        loss = supervised_loss + (alpha) * pc_loss
        self.student_model.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(self.student_model, self.teacher_model, 0.9, iteration)
        postfix_dict = {'loss': loss.item(),
                        'supervised_loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'target_inputs': target_inputs,
                            'source_labels': source_labels,
                            'target_labels': target_labels,
                            'inputstaug': inputstaug,
                            'outputs': outputs,
                            'outputst': outputst}
    
    def tensorboard_logging(self, tensorboard_dict):
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
            if inputstaug is not None:
                save_images(writer=self.writer, images=tensorboard_dict['inputstaug'], normalize=True, sigmoid=False,
                            iteration=self.iterations, name='inputsaug')
            if outputst is not None:
                save_images(writer=self.writer, images=tensorboard_dict['outputst'], normalize=False, sigmoid=True,
                            iteration=self.iterations, name='outputs_target')
                
    def load(self, checkpoint_path):
        self.starting_epoch = int(os.path.basename(checkpoint_path.split('.')[0]).split('_')[-1])
        checkpoint = torch.load(checkpoint_path)
        self.student_model = self.student_model.load_state_dict(checkpoint['student_model'])
        self.teacher_model = self.teacher_model.load_state_dict(checkpoint['teacher_model'])
        self.seg_optimizer = self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'student_model': self.student_model.state_dict(),
                    'teacher_model': self.teacher_model.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
