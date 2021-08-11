# this code come from: A2_slovenia_llr_inferenceSimilarity_seed2.py
import torch
import numpy as np
import os
import json
import pandas as pd
import argparse
import torch.optim as optim
import time
import sys
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict
import itertools

from utils import to_var_gpu
from utils import apply_transform
from utils import dice_soft_loss
from utils import generate_affine
from utils import loop_iterable
from utils import ss_loss
from utils import non_geometric_augmentations
from utils import load_default_config
from utils import update_ema_variables
from utils import bland_altman_loss
from utils import save_images

from networks import Destilation_student_matchingInstance, SplitHeadModel, GeneratorUnet, get_fpn
from functools import partial
import multiprocessing


from dataset import SliceDataset, WholeVolumeDataset, SliceDatasetTumour,\
WholeVolumeDatasetTumour, get_monai_slice_dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from paths import results_paths, ms_path, model_saving_paths, tensorboard_paths
from inference import inference_ms, inference_tumour, inference_crossmoda
from networks import DiscriminatorDomain, DiscriminatorCycleGAN, DiscriminatorCycleGANSimple
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
from models.cyclegan import CycleganModel
from models.mean_teacher import MeanTeacherModel
from models.supervised_joint import SupervisedJointModel
from models.supervised import SupervisedModel
from models.ada import ADAModel

def train(args, model, 
          source_train_slice_dataset, source_val_dataset, source_test_dataset,
          target_train_slice_dataset, source_val_slice_dataset, target_val_slice_dataset,
          target_val_dataset, target_test_dataset):
    source_dl = loop_iterable(torch.utils.data.DataLoader(source_train_slice_dataset,
                                                          batch_size=args.batch_size, shuffle=True))
    target_dl = loop_iterable(torch.utils.data.DataLoader(target_train_slice_dataset,
                                                          batch_size=args.batch_size, shuffle=True))
    source_val_slice_dl = loop_iterable(torch.utils.data.DataLoader(source_val_slice_dataset,
                                                                    batch_size=args.batch_size, shuffle=True))
    target_val_slice_dl = loop_iterable(torch.utils.data.DataLoader(target_val_slice_dataset,
                                                                    batch_size=args.batch_size, shuffle=True))
    
    epoch = -1
    iteration = 0
    is_training = True
    model.initialise()
    try:
        while is_training:
            epoch += 1
            start_time = time.time()
            train_mini_batch_indices = np.arange(0, len(source_train_slice_dataset), args.batch_size)
            torch.manual_seed(epoch)
            model.epoch_reset()
            # Training loop
            with tqdm(total=len(train_mini_batch_indices), file=sys.stdout) as pbar:
                for indb, _ in enumerate(train_mini_batch_indices):                   
                    pbar.update(1)
                    iteration = epoch * len(train_mini_batch_indices) + indb
                    model.iterations = iteration
                    postfix_dict, tensorboard_dict = model.training_loop(source_dl, target_dl)
                    pbar.set_postfix(postfix_dict)
                    if iteration % args.tensorboard_every_n == 0:
                        model.tensorboard_logging(postfix_dict=postfix_dict, tensorboard_dict=tensorboard_dict, split='train')
                    if iteration % args.save_every_n == 0:
                        model.save()
                    if iteration >= args.iterations:
                        print('Training ending!')
                        print('SAVING MODEL')
                        model.save()
                        is_training = False
                        break
            # Validation loop
            print('Validation!')
            val_mini_batch_indices = np.arange(0, len(source_val_slice_dataset), args.batch_size)
            running_postfix_dict  = {}
            model.epoch_reset()
            with tqdm(total=len(val_mini_batch_indices), file=sys.stdout) as pbar:
                for indb, _ in enumerate(val_mini_batch_indices):
                    pbar.update(1)
                    postfix_dict, tensorboard_dict = model.validation_loop(source_dl, target_dl)
                    if not running_postfix_dict:
                        running_postfix_dict = postfix_dict
                    else:
                        # Update postfix dict using moving average formula
                        for key, postfix_value in postfix_dict.items():
                            running_postfix_dict[key] = (postfix_value + indb*running_postfix_dict[key] )/ (indb + 1)
                    pbar.set_postfix(running_postfix_dict)
            model.tensorboard_logging(postfix_dict=running_postfix_dict,
                                      tensorboard_dict=tensorboard_dict, split='val')
            end_time = time.time()
            time_epoch = (end_time - start_time) / 60
            print('Time: {}'.format(time_epoch))
    except KeyboardInterrupt:
        print('Interrupted training at iteration {}'.format(iteration))
        print('SAVING MODEL')
        model.save()
    model.writer.close()


def main(args):
    
    band = 'refactor_{}_{}_{}'.format(args.method, args.source, args.target)
    assert args.task in ['ms', 'tumour', 'crossmoda']
    # Directory paths
    models_folder = model_saving_paths[os.uname().nodename]
    results_folder = results_paths[os.uname().nodename]
    tensorboard_folder = tensorboard_paths[os.uname().nodename]
    run_name='{}_{}'.format(band, args.tag)
    model_factory = {'cyclegan': CycleganModel, 'ada': ADAModel, 'mean_teacher': MeanTeacherModel,
                     'supervised_joint': SupervisedJointModel, 'supervised': SupervisedModel}
    slice_dataset = {'ms': SliceDataset, 'crossmoda': get_monai_slice_dataset, 'tumour': SliceDatasetTumour}[args.task]
    whole_volume_dataset = {'ms': WholeVolumeDataset, 'crossmoda': WholeVolumeDataset, 'tumour': WholeVolumeDatasetTumour}[args.task]
    source_train_slice_dataset = slice_dataset(ms_path[os.uname().nodename][args.source],
                                         exclude_slices = list(range(70,192)) + list(range(20)),
                                         paddtarget=args.paddtarget, split='train', tumour_only=bool(args.tumour_only),
                                         slice_selection_method='mask', dataset_split_csv=args.source_split)
    source_val_slice_dataset = slice_dataset(ms_path[os.uname().nodename][args.source],
                                         paddtarget=args.paddtarget, split='val', tumour_only=bool(args.tumour_only),
                                         slice_selection_method='mask', dataset_split_csv=args.source_split)
    source_val_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.source + '_whole'], split='val',
                                              tumour_only=bool(args.tumour_only), paddtarget=args.paddtarget,
                                              dataset_split_csv=args.source_split)
    source_test_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.source + '_whole'], split='test',
                                               tumour_only=bool(args.tumour_only), paddtarget=args.paddtarget,
                                               dataset_split_csv=args.source_split)
    target_train_slice_dataset = slice_dataset(ms_path[os.uname().nodename][args.target],
                                         paddtarget=args.paddtarget, split='train',
                                         exclude_slices = list(range(70,192)) + list(range(20)),
                                         slice_selection_method='mask', dataset_split_csv=args.target_split)
    target_val_slice_dataset = slice_dataset(ms_path[os.uname().nodename][args.target],
                                             paddtarget=args.paddtarget, split='val',
                                             slice_selection_method='mask', dataset_split_csv=args.target_split)
    target_train_whole_vol_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.target + '_whole'],
                                                          split='train', paddtarget=args.paddtarget,
                                                          dataset_split_csv=args.target_split)
    target_val_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.target + '_whole'], split='val',
                                              paddtarget=args.paddtarget,
                                              dataset_split_csv=args.target_split)
    writer = SummaryWriter(tensorboard_folder+'/{}/{}_{}'.format(args.task, band, args.tag))
    inference_func = {'ms': inference_ms, 'tumour': inference_tumour, 'crossmoda': inference_crossmoda}[args.task]
#     save_images = partial(save_images, k=1 if args.task == 'tumour' else 3)
    model = model_factory[args.method](cf=args, writer=writer, models_folder=models_folder,
                                       results_folder=results_folder, tensorboard_folder=tensorboard_folder, run_name=run_name)
    if args.checkpoint != "null":
        print('Loading model from checkpoint')
        print(args.checkpoint)
        model.load(args.checkpoint)
    if args.infer:
        p = multiprocessing.Pool(4)
        target_test_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.target + '_whole'], split='test',
                                                   tumour_only=bool(args.tumour_only), paddtarget=args.paddtarget,
                                                   dataset_split_csv=args.target_split)
        performance_target_train_s, performance_target_train, _, _ = \
            inference_func(args, p, seg_model, target_test_dataset, prefix=os.path.join(results_folder, 'target_test'),
                           epoch=starting_epoch)  # hack
    else:
        train(args=args, model=model, source_val_slice_dataset=source_val_slice_dataset, target_val_slice_dataset=target_val_slice_dataset,
              source_train_slice_dataset=source_train_slice_dataset,
              target_train_slice_dataset=target_train_slice_dataset,
              source_test_dataset=source_test_dataset, target_test_dataset=target_train_whole_vol_dataset,  # hack
              source_val_dataset=source_val_dataset, target_val_dataset=target_val_dataset)

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='MICCAI2020')
    parser.add_argument('--lr', type=float, metavar='LR', help='learning rate (default: (1e-4))')
    parser.add_argument('--labels', type=int, metavar='LABELS', help='number of labels (default: 1)')
    parser.add_argument('--channels', type=int, metavar='CHANNELS', help='number of channels (default: 1)')
    parser.add_argument('--iterations', type=int, metavar='ITERATIONS', help='number of iterations to train')
    parser.add_argument('--diceThs', type=float, metavar='DICETHS', help='Threshold for dice estimation')
    parser.add_argument('--batch_size', type=int, metavar='BATCHSIZE', help='batch size')
    parser.add_argument('--paddsource', type=int, metavar='PADD', help='PADD')
    parser.add_argument('--paddtarget', type=int, metavar='PADD', help='PADD')
    # parser.add_argument('--SaveTrLoss',  type = int , )
    parser.add_argument('--affine_rot_degree', type=float, metavar='AffineRot', help='Affine Rotations parameter')
    parser.add_argument('--affine_scale', type=float, metavar='AffineScale', help='Affine scale parameter')
    parser.add_argument('--affine_shearing', type=float, metavar='AffineShearing',
                        help='Affine shearing scale')
    parser.add_argument('--loss', type=str, metavar='LOSS', help='Loss, dice or bland_altman')
    parser.add_argument('--iterations_adapt', type=int, metavar='ITERATIONSADAPT',
                        help='This is the iteration count where the adaptation start')
    parser.add_argument('--thssaving', type=float, metavar='THSSAVING', help='ths to save model')
    parser.add_argument('--thstesting', type=float, metavar='THSTESTING', help='ths to select testing slices')
    parser.add_argument('--alpha_lweights', type=float, metavar='ALPHA', help='alpha weights the pc loss')
    parser.add_argument('--beta_lweights', type=float, metavar='BETA', help='beta weights the adversarial loss')
    parser.add_argument('--source', type=str, metavar='Data', help='data name')
    parser.add_argument('--target', type=str, metavar='Data', help='data name')
    parser.add_argument('--method', type=str, metavar='METHODS', help='method name')
    parser.add_argument('--tag', type=str, metavar='TAG', help='Experiment tag')
    parser.add_argument('--task', type=str, metavar='INPUT', help='wmh or tumour')
    parser.add_argument('--source_split', type=str, help='path to dataset_split.csv for source')
    parser.add_argument('--target_split', type=str, help='path to dataset_split.csv for target')
    parser.add_argument('--save_every_n', type=int)
    parser.add_argument('--tensorboard_every_n', type=int)
    parser.add_argument('--use_fixmatch', type=int)
    parser.add_argument('--config', type=str, help='path to json file, will override all other config')
    parser.add_argument('--discriminator_complexity', type=float,
                        help='8*complexity gives the number of initial layers')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--infer', type=int, help='0 if training else 1')
    parser.add_argument('--tumour_only', type=int, help='1 if tumour only labels to be used else 0')
    args = parser.parse_args()
    config = load_default_config(args.task) if args.config is None else json.load(open(args.config, 'r'))
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        arg_dict[key] = config[key] if value is None else value
    # hostname_dir = {'dgx1-1': '/raid/tomvars', 'pretzel': '/raid/tom', 'bd0795ec38f7': '/data2/tom'}
    # if args.checkpoint != "null" and args.checkpoint is not None:
    #     args.checkpoint = args.checkpoint.replace('/raid/tomvars', hostname_dir[os.uname().nodename])
    main(args=args)