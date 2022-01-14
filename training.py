# this code come from: A2_slovenia_llr_inferenceSimilarity_seed2.py
import torch
import numpy as np
import os
import json
import pandas as pd
import argparse
import time
from pathlib import Path
import sys
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict
import itertools
from functools import partial
import multiprocessing
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from monai.losses.dice import DiceLoss
from monai.data.nifti_saver import NiftiSaver
from monai.data import DataLoader, ThreadDataLoader

from inference import patch_based_inference, slice_based_inference
from src.utils import to_var_gpu
from src.utils import apply_transform
from src.utils import dice_soft_loss
from src.utils import generate_affine
from src.utils import loop_iterable
from src.utils import ss_loss
from src.utils import non_geometric_augmentations
from src.utils import load_default_config
from src.utils import update_ema_variables
from src.utils import bland_altman_loss
from src.utils import save_images
from src.networks import Destilation_student_matchingInstance, SplitHeadModel, GeneratorUnet, get_fpn
from src.networks import DiscriminatorDomain, DiscriminatorCycleGAN, DiscriminatorCycleGANSimple
from src.dataset import get_monai_slice_dataset, infer_on_subject, get_monai_patch_dataset
from src.paths import results_paths, data_paths, model_saving_paths, tensorboard_paths, inference_paths
from src.cyclegan import CycleganModel
from src.mean_teacher import MeanTeacherModel
from src.supervised_joint import SupervisedJointModel
from src.supervised import SupervisedSegmentation3DModel
from src.ada import ADAModel
from src.ada_3d import ADA3DModel
from src.ada_3d_dyn_unet import ADA3DDynUNETModel
from src.icmsc import ICMSCModel
from src.cycada import CycadaModel
from src.supervised_retina_unet import SupervisedRetinaUNetModel
from src.supervised_retina_unet_3d import SupervisedRetinaUNet3DModel
from src.supervised_fcos_3d import SupervisedFCOS3DModel
from src.ada_retina_unet import AdaRetinaUNetModel
from src.ada_retina_unet_3d import AdaRetinaUNet3DModel


def train(args, model, starting_iteration,
          source_train_dataset,
          target_train_dataset,
          source_val_dataset,
          target_val_dataset):
#     source_dl = loop_iterable(DataLoader(source_train_dataset, batch_size=args.batch_size,
#                                          shuffle=True, collate_fn=lambda x: x, drop_last=True, num_workers=4))
#     target_dl = loop_iterable(DataLoader(target_train_dataset, batch_size=args.batch_size,
#                                          shuffle=True, collate_fn=lambda x: x, drop_last=True, num_workers=4))
#     source_val_dl = loop_iterable(DataLoader(source_val_dataset, batch_size=args.batch_size,
#                                                    shuffle=True, collate_fn=lambda x: x, drop_last=True, num_workers=4))
#     target_val_dl = loop_iterable(DataLoader(target_val_dataset, batch_size=args.batch_size,
#                                                    shuffle=True, collate_fn=lambda x: x, drop_last=True, num_workers=4))
    source_dl = loop_iterable(ThreadDataLoader(source_train_dataset, batch_size=args.batch_size,
                                         shuffle=True, collate_fn=lambda x: x, drop_last=False, num_workers=4, buffer_size=40,
                                              persistent_workers=True, pin_memory=False))
    target_dl = loop_iterable(ThreadDataLoader(target_train_dataset, batch_size=args.batch_size,
                                         shuffle=True, collate_fn=lambda x: x, drop_last=False, num_workers=1, buffer_size=14))
    source_val_dl = loop_iterable(ThreadDataLoader(source_val_dataset, batch_size=args.batch_size,
                                                   shuffle=True, collate_fn=lambda x: x, drop_last=False, num_workers=4, buffer_size=40,
                                                  persistent_workers=True, pin_memory=False))
    target_val_dl = loop_iterable(ThreadDataLoader(target_val_dataset, batch_size=args.batch_size,
                                                   shuffle=True, collate_fn=lambda x: x, drop_last=False, num_workers=1, buffer_size=2))
    epoch = -1
    iteration = starting_iteration
    is_training = True
    model.initialise()
    try:
        while is_training:
            epoch += 1
            start_time = time.time()
            train_mini_batch_indices = np.arange(0, len(source_train_dataset), args.batch_size)
            torch.manual_seed(epoch)
            model.epoch_reset()
            # Training loop
            with tqdm(total=len(train_mini_batch_indices), file=sys.stdout) as pbar:
                for indb, _ in enumerate(train_mini_batch_indices):
                    if indb >= args.max_iterations_per_epoch:
                        print(f'Stopping epoch at {indb} iterations')
                        break
                    pbar.update(1)
                    iteration = epoch * len(train_mini_batch_indices) + indb
                    model.iterations = iteration
                    postfix_dict, tensorboard_dict = model.training_loop(source_dl, target_dl)
                    if postfix_dict is None and tensorboard_dict is None:
                        continue
                    pbar.set_postfix(postfix_dict)
                    if iteration % args.tensorboard_every_n == 0:
                        print('Got here?')
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
            val_mini_batch_indices = np.arange(0, len(source_val_dataset), args.batch_size)
            running_postfix_dict  = {}
            model.epoch_reset()
            with tqdm(total=len(val_mini_batch_indices), file=sys.stdout) as pbar:
                for indb, _ in enumerate(val_mini_batch_indices):
                    pbar.update(1)
                    with torch.no_grad():
                        postfix_dict, val_tensorboard_dict = model.validation_loop(source_val_dl, target_val_dl)
                    if not running_postfix_dict:
                        running_postfix_dict = postfix_dict
                    else:
                        # Update postfix dict using moving average formula
                        for key, postfix_value in postfix_dict.items():
                            running_postfix_dict[key] = (postfix_value + indb*running_postfix_dict[key] )/ (indb + 1)
                    pbar.set_postfix(running_postfix_dict)
            if epoch % args.val_tensorboard_every_n_epochs == 0:
                model.tensorboard_logging(postfix_dict=running_postfix_dict,
                                          tensorboard_dict=val_tensorboard_dict, split='val')
            end_time = time.time()
            time_epoch = (end_time - start_time) / 60
            print('Time: {}'.format(time_epoch))
    except KeyboardInterrupt:
        print('Interrupted training at iteration {}'.format(iteration))
        print('SAVING MODEL')
        model.save()
    model.writer.close()

def infer(args, model, inference_dir):
    dataset_split_df = pd.read_csv(args.inference_split, names=['subject_id', 'split'])
#     data_dir = data_paths[os.uname().nodename][args.target] # Target data
    data_dir = '/data2/tom/crossmoda/target_validation/slices'
    flair_filenames = os.listdir(os.path.join(data_dir, 'flair'))
    subject_ids = [x.split('_slice')[0] for x in flair_filenames]
    slice_idx_arr = [int(x.split('_')[3].replace('.nii.gz', '')) for x in flair_filenames]
    label_paths = [os.path.join(data_dir, 'labels', x.replace('FLAIR', 'wmh')) for x in flair_filenames]
    flair_paths = [os.path.join(data_dir, 'flair', x) for x in flair_filenames]
    files_df = pd.DataFrame(
        data=[(subj, slice_idx, fp, lp) for subj, slice_idx, fp, lp in zip(subject_ids, slice_idx_arr,
                                                                           label_paths, flair_paths)],
        columns=['subject_id', 'slice_index', 'label_path', 'flair_path']
    )
    # Need to loop over subject ids
    subject_ids = files_df.subject_id.unique()
    with tqdm(total=len(subject_ids), file=sys.stdout) as pbar:
        for subject_id in subject_ids:
            output_path = str(Path(inference_dir) / subject_id)
            print(output_path)
            whole_volume_path = str(Path(data_dir).parent / 'whole' / 'flair' / (subject_id + '.nii.gz'))
            print(whole_volume_path)
            infer_on_subject(model, output_path, whole_volume_path, files_df, subject_id, batch_size=10)
            pbar.update(1)

    
def main(args):
    
    band = 'refactor_{}_{}_{}'.format(args.method, args.source, args.target)
    # Directory paths
    models_folder = model_saving_paths[os.uname().nodename]
    results_folder = results_paths[os.uname().nodename]
    tensorboard_folder = os.path.join(tensorboard_paths[os.uname().nodename], args.task)
    inference_folder = inference_paths[os.uname().nodename]
    run_name='{}_{}'.format(band, args.tag)
    model_factory = {'cyclegan': CycleganModel,
                     'ada': ADAModel,
                     'ada_3d': ADA3DModel,
                     'ada_3d_dyn_unet': ADA3DDynUNETModel,
                     'mean_teacher': MeanTeacherModel,
                     'supervised_joint': SupervisedJointModel,
                     'supervised': SupervisedSegmentation3DModel,
                     'icmsc': ICMSCModel, 'cycada': CycadaModel,
                     'supervised_retina_unet': SupervisedRetinaUNetModel,
                     'supervised_retina_unet_3d': SupervisedRetinaUNet3DModel,
                     'supervised_fcos_3d': SupervisedFCOS3DModel,
                     'ada_retina_unet': AdaRetinaUNetModel,
                     'ada_retina_unet_3d': AdaRetinaUNet3DModel
                    } # If you want to go 3D implement a new model...
    
    dataset_factory = {2: get_monai_slice_dataset,
                       3: get_monai_patch_dataset}[args.dims]

    # Customise the data input pipeline based on the current task or model used #
    label_mapping = {
                     'crossmoda': {0: 0, 1: 0, 2: 1},
                     'ms': {0: 0, 1: 1, 2: 1, 3: 1},
                     'microbleed': {0: 0, 1: 1, 2: 1, 3: 1},
                     'tumour': {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}
                    }.get(args.data_task, None)
    bboxes = True if args.task == 'object_detection' else False
    return_aug = True if args.method.startswith('ada') else False # If True, return augmented inputs
    #############################################################################
    source_train_dataset = dataset_factory(data_paths[os.uname().nodename][args.source], cf=args,
                                           exclude_slices = [], #list(range(70,192)) + list(range(20)),
                                           training_aug=args.training_aug,
                                           include_dist_map=args.use_boundary_loss,
                                           spatial_dims=args.dims,
                                           spatial_size=args.spatial_size, split='train',
                                           dataset_split_csv=args.source_split,
                                           bounding_boxes=bboxes, return_aug=False, label_mapping=label_mapping)
    source_val_dataset = dataset_factory(data_paths[os.uname().nodename][args.source], cf=args,
                                         training_aug=False,
                                         include_dist_map=args.use_boundary_loss,
                                         spatial_dims=args.dims,
                                         spatial_size=args.spatial_size, split='val', dataset_split_csv=args.source_split,
                                         bounding_boxes=bboxes, return_aug=False, label_mapping=label_mapping)
    target_train_dataset = dataset_factory(data_paths[os.uname().nodename][args.target], cf=args,
                                           training_aug=False,
                                           include_dist_map=args.use_boundary_loss,
                                           spatial_size=args.spatial_size, split='train',
                                           exclude_slices = [], dataset_split_csv=args.target_split,
                                           bounding_boxes=bboxes, return_aug=return_aug, label_mapping=label_mapping)
    target_val_dataset = dataset_factory(data_paths[os.uname().nodename][args.target], cf=args,
                                         training_aug=False,
                                         include_dist_map=args.use_boundary_loss,
                                         spatial_size=args.spatial_size, split='val', dataset_split_csv=args.target_split,
                                         bounding_boxes=bboxes, return_aug=return_aug, label_mapping=label_mapping)
    writer = SummaryWriter(tensorboard_folder+'/{}/{}_{}'.format(args.data_task, band, args.tag))
    inference_func = {2: slice_based_inference, 3: patch_based_inference}[args.dims]
    model = model_factory[args.method](cf=args, writer=writer,
                                       models_folder=models_folder,
                                       results_folder=results_folder,
                                       tensorboard_folder=tensorboard_folder,
                                       run_name=run_name)
    if args.checkpoint != "null":
        print('Loading model from checkpoint')
        print(args.checkpoint)
        model.load(args.checkpoint)
        starting_iteration = int(args.checkpoint.split('_')[-1].replace('.pt', ''))
    else:
        starting_iteration = 0
    if args.infer:
        source_infer_dataset, target_infer_dataset = None, None
        if args.source_inference_split != "null":
            source_infer_dataset = dataset_factory(data_paths[os.uname().nodename][args.source], cf=args,
                                                   spatial_size=args.spatial_size, split='infer',
                                                   dataset_split_csv=args.source_inference_split,
                                                   bounding_boxes=False, return_aug=False, label_mapping=label_mapping)
        if args.target_inference_split != "null":
            target_infer_dataset = dataset_factory(data_paths[os.uname().nodename][args.target], cf=args,
                                                   spatial_size=args.spatial_size, split='infer',
                                                   dataset_split_csv=args.target_inference_split,
                                                   bounding_boxes=False, return_aug=False, label_mapping=label_mapping)
        inference_func(args=args, model=model,
                       source_dataset=source_infer_dataset,
                       target_dataset=target_infer_dataset,
                       inference_dir=os.path.join(inference_folder, run_name))
    else:
        train(args=args, model=model, starting_iteration=starting_iteration,
              source_val_dataset=source_val_dataset,
              target_val_dataset=target_val_dataset,
              source_train_dataset=source_train_dataset,
              target_train_dataset=target_train_dataset)

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='MICCAI2020')
    parser.add_argument('--lr', type=float, metavar='LR', help='learning rate (default: (1e-4))')
    parser.add_argument('--labels', type=int, metavar='LABELS', help='number of labels (default: 1)')
    parser.add_argument('--channels', type=int, metavar='CHANNELS', help='number of channels (default: 1)')
    parser.add_argument('--iterations', type=int, metavar='ITERATIONS', help='number of iterations to train')
    parser.add_argument('--diceThs', type=float, metavar='DICETHS', help='Threshold for dice estimation')
    parser.add_argument('--batch_size', type=int, metavar='BATCHSIZE', help='batch size')
    parser.add_argument('--dims', type=int, metavar='dim', help='Number of dimensions 2 or 3')
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
    parser.add_argument('--anchor_matching_strategy', type=str, help='atss or iou (only used in object_detection)')
    parser.add_argument('--bbox_loss', type=str, help='giou or l1 (only used in object_detection)')
    parser.add_argument('--source', type=str, metavar='Data', help='data name')
    parser.add_argument('--target', type=str, metavar='Data', help='data name')
    parser.add_argument('--method', type=str, metavar='METHODS', help='method name')
    parser.add_argument('--tag', type=str, metavar='TAG', help='Experiment tag')
    parser.add_argument('--data_task', type=str, metavar='INPUT', help=" 'ms', 'tumour' or 'crossmoda'")
    parser.add_argument('--task', type=str, metavar='INPUT', help="'object_detection' or 'segmentation'")
    parser.add_argument('--source_split', type=str, help='path to dataset_split.csv for source')
    parser.add_argument('--target_split', type=str, help='path to dataset_split.csv for target')
    parser.add_argument('--source_inference_split', type=str, help='path to dataset_split.csv for inference (split column is ignored)')
    parser.add_argument('--target_inference_split', type=str, help='path to dataset_split.csv for inference (split column is ignored)')
    parser.add_argument('--save_every_n', type=int)
    parser.add_argument('--max_iterations_per_epoch', type=int, default=10000)
    parser.add_argument('--tensorboard_every_n', type=int)
    parser.add_argument('--val_tensorboard_every_n_epochs', type=int, default=1)
    parser.add_argument('--use_fixmatch', type=int)
    parser.add_argument('--config', type=str, help='path to json file, will override all other config')
    parser.add_argument('--discriminator_complexity', type=float,
                        help='8*complexity gives the number of initial layers')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--infer', type=int, help='0 if training else 1')
    parser.add_argument('--tumour_only', type=int, help='1 if tumour only labels to be used else 0')
    parser.add_argument('--training_aug', type=int, help='1 if training_aug else 0')
    parser.add_argument('--use_boundary_loss', type=int, help='1 if use_boundary_loss else 0')
    parser.add_argument('--spatial_size', nargs='+', help='e.g 256 256')
    parser.add_argument('--spatial_crop_center', nargs='+', help='e.g 100, 138, 40')
    parser.add_argument('--spatial_crop_roi', nargs='+', help='e.g 168, 168, 80')
    parser.add_argument('--sampler', type=str, help='should be either "weighted" or "random"')
    args = parser.parse_args()
    config = load_default_config(args.data_task, args.dims) if args.config is None else json.load(open(args.config, 'r'))
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if key in config:
            arg_dict[key] = config[key] if value is None else value
    arg_dict['spatial_size'] = [int(f) for f in arg_dict['spatial_size']]
    # hostname_dir = {'dgx1-1': '/raid/tomvars', 'pretzel': '/raid/tom', 'bd0795ec38f7': '/data2/tom'}
    # if args.checkpoint != "null" and args.checkpoint is not None:
    #     args.checkpoint = args.checkpoint.replace('/raid/tomvars', hostname_dir[os.uname().nodename])
    main(args=args)
