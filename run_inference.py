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

from inference import inference_ms, inference_tumour, inference_crossmoda
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
from src.dataset import SliceDataset, WholeVolumeDataset, SliceDatasetTumour,\
WholeVolumeDatasetTumour, get_monai_slice_dataset, infer_on_subject
from src.paths import results_paths, ms_path, model_saving_paths, tensorboard_paths, inference_paths
from src.cyclegan import CycleganModel
from src.mean_teacher import MeanTeacherModel
from src.supervised_joint import SupervisedJointModel
from src.supervised import SupervisedModel
from src.ada import ADAModel
from src.icmsc import ICMSCModel
from src.cycada import CycadaModel

def infer(args, model, inference_dir):
    dataset_split_df = pd.read_csv(args.inference_split, names=['subject_id', 'split'])
    data_dir = ms_path[os.uname().nodename][args.target] # Target data
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
            whole_volume_path = str(Path(ms_path[os.uname().nodename][args.target]).parent / 'whole' / 'flair' / (subject_id + '.nii.gz'))
            print(whole_volume_path)
            infer_on_subject(model, output_path, whole_volume_path, files_df, subject_id, batch_size=10)
            pbar.update(1)

    
def main(args):
    
    band = 'refactor_{}_{}_{}'.format(args.method, args.source, args.target)
    assert args.task in ['ms', 'tumour', 'crossmoda']
    # Directory paths
    models_folder = model_saving_paths[os.uname().nodename]
    results_folder = results_paths[os.uname().nodename]
    tensorboard_folder = tensorboard_paths[os.uname().nodename]
    inference_folder = inference_paths[os.uname().nodename]
    run_name='{}_{}'.format(band, args.tag)
    model_factory = {'cyclegan': CycleganModel, 'ada': ADAModel, 'mean_teacher': MeanTeacherModel,
                     'supervised_joint': SupervisedJointModel, 'supervised': SupervisedModel,
                     'icmsc': ICMSCModel, 'cycada': CycadaModel}
    writer = SummaryWriter(tensorboard_folder+'/{}/{}_{}'.format(args.task, band, args.tag))
    model = model_factory[args.method](cf=args, writer=writer, models_folder=models_folder,
                                       results_folder=results_folder, tensorboard_folder=tensorboard_folder, run_name=run_name)
    if args.checkpoint != "null":
        print('Loading model from checkpoint')
        print(args.checkpoint)
        model.load(args.checkpoint)
    if args.infer:
        infer(args=args, model=model, inference_dir=os.path.join(inference_folder, run_name))
    else:
        raise Exception('This is the inference script!')

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
    parser.add_argument('--inference_split', type=str, help='path to dataset_split.csv for inference (split column is ignored)')
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
