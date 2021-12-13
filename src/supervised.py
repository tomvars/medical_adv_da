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

class SupervisedSegmentation3DModel(BaseModel):
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
        self.supervised_loss = DiceLoss(softmax=True)
        self.dice_ce_loss = DiceCELoss(softmax=True, to_onehot_y=True)
        self.iterations = self.cf.iterations

    def initialise(self):
        self.seg_model.cuda()
        
    def training_loop(self, source_dl, target_dl):
        source_batch = next(source_dl)
        source_inputs, source_labels = (torch.tensor(np.stack([f[0]['inputs'] for f in source_batch])).to(self.device),
                                        torch.tensor(np.stack([f[0]['labels'] for f in source_batch])).to(self.device))
        self.seg_model.train()
        source_outputs, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)        
        supervised_loss = self.dice_ce_loss(source_outputs, source_labels)
        self.seg_optimizer.zero_grad()
        supervised_loss.backward()
        self.seg_optimizer.step()
        
        postfix_dict = {'loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'source_labels': source_labels,
                            'source_outputs': source_outputs[:, 1:, ...]}
        return postfix_dict, tensorboard_dict
    
    def validation_loop(self, source_dl, target_dl):
        source_batch = next(source_dl)
        source_inputs, source_labels = (torch.tensor(np.stack([f[0]['inputs'] for f in source_batch])).to(self.device),
                                        torch.tensor(np.stack([f[0]['labels'] for f in source_batch])).to(self.device))
        self.seg_model.eval()
        source_outputs, _, _, _, _, _, _, _, _ = self.seg_model(source_inputs)
        supervised_loss = self.dice_ce_loss(source_outputs, source_labels)
        
        postfix_dict = {'loss': supervised_loss.item()}
        tensorboard_dict = {'source_inputs': source_inputs,
                            'source_labels': source_labels,
                            'source_outputs': source_outputs[:, 1:, ...]}
        return postfix_dict, tensorboard_dict
    
    def tensorboard_logging(self, postfix_dict, tensorboard_dict, split):
        if self.cf.data_task in ['ms_3d', 'microbleed_3d', 'crossmoda_3d', 'tumour']:
            save_images(writer=self.writer, images=tensorboard_dict['source_labels'], normalize=True, sigmoid=False,
                        iteration=self.iterations, name='source_labels/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['source_outputs'], normalize=False, sigmoid=True,
                        iteration=self.iterations, name='source_outputs/{}'.format(split), png=False)
            save_images(writer=self.writer, images=tensorboard_dict['source_inputs'], normalize=True,
                        sigmoid=False, png=False,
                        iteration=self.iterations, name='source_inputs/{}'.format(split))
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
        self.seg_model.load_state_dict(checkpoint['seg_model'])
        self.seg_optimizer.load_state_dict(checkpoint['seg_optimizer'])
    
    def save(self):
        torch.save({'seg_model': self.seg_model.state_dict(),
                    'seg_optimizer': self.seg_optimizer.state_dict(),
                   }, os.path.join(self.models_folder, self.run_name + '_{}.pt'.format(self.iterations)))
        
    def epoch_reset(self):
        pass
    
    def inference_func(self, inference_inputs):
        with torch.no_grad():
            inference_outputs, _, _, _, _, _, _, _, _ = self.seg_model(inference_inputs)
        return (torch.sigmoid(inference_outputs[:, 1:, ...]) > 0.5).float()