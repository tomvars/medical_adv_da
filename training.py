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

from models import Destilation_student_matchingInstance, SplitHeadModel, GeneratorUnet, get_fpn
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


def train(args, inference_func, model, optimizer, criterion, criterion2, scheduler, writer, source_train_dataset,
          source_val_dataset, save_images, source_test_dataset, target_train_dataset, model_t,
          discriminator, optimizer_discriminator, results_folder, source_val_slice_dataset, target_val_slice_dataset,
          target_val_dataset, target_test_dataset, model_folder, run_name, cross_moda_args):
    source_dl = loop_iterable(torch.utils.data.DataLoader(source_train_dataset,
                                                          batch_size=args.batch_size, shuffle=True))
    target_dl = loop_iterable(torch.utils.data.DataLoader(target_train_dataset,
                                                          batch_size=args.batch_size, shuffle=True))
    source_val_slice_dl = loop_iterable(torch.utils.data.DataLoader(source_val_slice_dataset,
                                                          batch_size=args.batch_size, shuffle=True))
    target_val_slice_dl = loop_iterable(torch.utils.data.DataLoader(target_val_slice_dataset,
                                                                    batch_size=args.batch_size, shuffle=True))

    p = multiprocessing.Pool(10)
    epoch = -1
    iteration = 0
    is_training = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    if cross_moda_args is not None:
        generator_s_t, \
        generator_t_s, \
        discriminator_s, \
        discriminator_t, optimizer_discriminator, optimizer_generator = cross_moda_args
        criterion_cycle = torch.nn.L1Loss()
        gan_loss = torch.nn.CrossEntropyLoss()
        spatial_discriminator_loss = torch.nn.MSELoss()
    try:
        while is_training:
            epoch += 1
            start_time = time.time()
            scheduler.step()
            indxMiniBatch = np.arange(0, len(source_train_dataset), args.batch_size)
            running_loss = 0.0
            torch.manual_seed(epoch)
            correct = 0
            correct_s = 0
            correct_t = 0
            num_of_subjects = 0
            with tqdm(total=len(indxMiniBatch), file=sys.stdout) as pbar:
                for indb, batch in enumerate(indxMiniBatch):
                    if iteration < args.iterations_adapt:
                        alpha = 0
                        beta = 0
                    else:
                        alpha = args.alpha_lweights
                        beta = args.beta_lweights
                    model.train()
                    source_batch = next(source_dl)
                    source_inputs, source_labels = (source_batch['inputs'].to(device),
                                                    source_batch['labels'].to(device))
                    target_batch = next(target_dl)
                    target_inputs, target_labels = (target_batch['inputs'].to(device),
                                                    target_batch['labels'].to(device))
                    inputstaug = None
                    outputst = None
                    loss_fn = DiceLoss(to_onehot_y=True)
                    if args.method == 'supervised_joint':

                        batch_source = source_inputs.cpu().numpy()
                        batch_source = p.map(
                            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                            np.copy(batch_source))
                        batch_source = p.map(
                            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                            np.copy(batch_source))
                        inputstaug = torch.Tensor(batch_source).cuda()
                        theta_source, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                                          scale=args.affine_scale,
                                                          shearingScale=args.affine_shearing)
                        inputstaug = apply_transform(inputstaug, theta_source)
                        outputs, _, _, _, _, _, _, _, _, _ = model(inputstaug)

                        batch_target = target_inputs.cpu().numpy()
                        batch_target = p.map(
                            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                            np.copy(batch_target))
                        batch_target = p.map(
                            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                            np.copy(batch_target))
                        inputstaug = torch.Tensor(batch_target).cuda()
                        theta_target, Theta_inv = generate_affine(inputstaug, degreeFreedom=args.affine_rot_degree,
                                                          scale=args.affine_scale,
                                                          shearingScale=args.affine_shearing)
                        inputstaug = apply_transform(inputstaug, theta_target)
                        outputst, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        source_labels_transformed = apply_transform(source_labels, theta_source)
                        target_labels_transformed = apply_transform(target_labels, theta_target)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels_transformed) +\
                                          dice_soft_loss(torch.sigmoid(outputst), target_labels_transformed)
                        pc_loss = torch.tensor(0.0)
                        loss = supervised_loss
                        
                    elif args.method == 'cyclegan':
                        # Training CycleGAN
                        # model.eval()
                        discriminator_s.train()
                        discriminator_t.train()
                        generator_s_t.train()
                        generator_t_s.train()
                        generated_target = generator_s_t(source_inputs)
                        generated_source = generator_t_s(target_inputs)
                        cycle_source = generator_t_s(generated_target)
                        cycle_target = generator_s_t(generated_source)
                        cycle_loss_source = criterion_cycle(source_inputs, cycle_source)
                        cycle_loss_target = criterion_cycle(target_inputs, cycle_target)
                        cycle_loss = cycle_loss_source + cycle_loss_target
                        preds_discriminator_s = discriminator_s(torch.cat([generated_source, source_inputs]))
                        preds_discriminator_t = discriminator_t(torch.cat([target_inputs, generated_target]))
                        if len(preds_discriminator_s.shape) == 2:
                            labels_discriminator_s = to_var_gpu(
                                torch.cat((torch.zeros(generated_source.size(0)),
                                           torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
                            labels_discriminator_t = to_var_gpu(
                                torch.cat((torch.zeros(generated_target.size(0)),
                                           torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
                            loss_discriminator_s = gan_loss(preds_discriminator_s, labels_discriminator_s)
                            loss_discriminator_t = gan_loss(preds_discriminator_t, labels_discriminator_t)
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
                            loss_discriminator_s = spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
                            loss_discriminator_t = spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)
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
                            discriminator_s.eval()
                            discriminator_t.eval()
                            generator_s_t.train()
                            generator_t_s.train()
                            loss = cycle_loss - loss_discriminator_s - loss_discriminator_t
                            optimizer_generator.zero_grad()
                            loss.backward()
                            optimizer_generator.step()
                            
                            discriminator_s.train()
                            discriminator_t.train()
                            generator_s_t.train()
                            generator_t_s.train()
                            preds_discriminator_s = discriminator_s(torch.cat(
                                [generated_source.detach(), source_inputs]))
                            preds_discriminator_t = discriminator_t(torch.cat(
                                [target_inputs, generated_target.detach()]))
                            if len(preds_discriminator_s.shape) == 2:
                                labels_discriminator_s = to_var_gpu(
                                    torch.cat((torch.zeros(generated_source.size(0)),
                                               torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
                                labels_discriminator_t = to_var_gpu(
                                    torch.cat((torch.zeros(generated_target.size(0)),
                                               torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
                                loss_discriminator_s = gan_loss(preds_discriminator_s, labels_discriminator_s)
                                loss_discriminator_t = gan_loss(preds_discriminator_t, labels_discriminator_t)
                            else: # Spatial discriminator!
                                preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
                                labels_discriminator_s = to_var_gpu(
                                    torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                                               torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
                                preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
                                labels_discriminator_t = to_var_gpu(
                                    torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                                               torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
                                loss_discriminator_s = spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
                                loss_discriminator_t = spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)

                            optimizer_discriminator.zero_grad()
                            loss_discriminator_s.backward()
                            loss_discriminator_t.backward()
                            optimizer_discriminator.step()
                        elif (acc_discriminator_s < 0.70 or acc_discriminator_t < 0.70):
                            discriminator_s.train()
                            discriminator_t.train()
                            generator_s_t.eval()
                            generator_t_s.eval()
                            preds_discriminator_s = discriminator_s(torch.cat(
                                [generated_source.detach(), source_inputs]))
                            preds_discriminator_t = discriminator_t(torch.cat(
                                [target_inputs, generated_target.detach()]))
                            if len(preds_discriminator_s.shape) == 2:
                                labels_discriminator_s = to_var_gpu(
                                    torch.cat((torch.zeros(generated_source.size(0)),
                                               torch.ones(source_inputs.size(0))), 0).type(torch.LongTensor))
                                labels_discriminator_t = to_var_gpu(
                                    torch.cat((torch.zeros(generated_target.size(0)),
                                               torch.ones(target_inputs.size(0))), 0).type(torch.LongTensor))
                                loss_discriminator_s = gan_loss(preds_discriminator_s, labels_discriminator_s)
                                loss_discriminator_t = gan_loss(preds_discriminator_t, labels_discriminator_t)
                            else: # Spatial discriminator!
                                preds_discriminator_s = preds_discriminator_s.view(-1) # 2 * batch_size * 7 * 7
                                labels_discriminator_s = to_var_gpu(
                                    torch.cat((torch.zeros(int(preds_discriminator_s.size(0)*0.5)),
                                               torch.ones(int(preds_discriminator_s.size(0)*0.5))), 0).type(torch.FloatTensor))
                                preds_discriminator_t = preds_discriminator_t.view(-1) # 2 * batch_size * 7 * 7
                                labels_discriminator_t = to_var_gpu(
                                    torch.cat((torch.zeros(int(preds_discriminator_t.size(0)*0.5)),
                                               torch.ones(int(preds_discriminator_t.size(0)*0.5))), 0).type(torch.FloatTensor))
                                loss_discriminator_s = spatial_discriminator_loss(preds_discriminator_s, labels_discriminator_s)
                                loss_discriminator_t = spatial_discriminator_loss(preds_discriminator_t, labels_discriminator_t)
                            optimizer_discriminator.zero_grad()
                            loss_discriminator_s.backward()
                            loss_discriminator_t.backward()
                            optimizer_discriminator.step()
                        else:
                            discriminator_s.eval()
                            discriminator_t.eval()
                            generator_s_t.train()
                            generator_t_s.train()
                            loss = cycle_loss - loss_discriminator_s - loss_discriminator_t
                            optimizer_generator.zero_grad()
                            loss.backward()
                            optimizer_generator.step()
                    elif args.method == 'supervised':
                        # Training segmentation model from Generated T2s
                        outputs, outputs2, _, _, _, _, _, _, _, _, _  = model(source_inputs)
                        tumour_labels = torch.where(source_labels == 1,
                                                    torch.ones_like(source_labels),
                                                    torch.zeros_like(source_labels))
                        cochlea_labels = torch.where(source_labels == 2,
                                                     torch.ones_like(source_labels),
                                                     torch.zeros_like(source_labels))
                        print(torch.sigmoid(outputs).sum())
                        print(torch.sigmoid(outputs2).sum())
                        print(tumour_labels.sum())
                        print(cochlea_labels.sum())
                        tumour_loss = dice_soft_loss(torch.sigmoid(outputs), tumour_labels)
                        cochlea_loss = dice_soft_loss(torch.sigmoid(outputs2), cochlea_labels)
                        supervised_loss = (tumour_loss + cochlea_loss) / 2.0
                        model.zero_grad()
                        supervised_loss.backward()
                        optimizer.step()
                        
                    elif args.method == 'A1':
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        outputst, _, _, _, _, _, _, _, _, _ = model(target_inputs)
                        outputst2, _, _, _, _, _, _, _, _, _ = model(target_inputs)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        pc_loss = alpha * criterion(torch.sigmoid(outputst), torch.sigmoid(outputst2))
                        loss = supervised_loss + pc_loss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                    elif args.method == 'A2':
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        outputst, _, _, _, _, _, _, _, _, _ = model(target_inputs)
                        Theta, Theta_inv = generate_affine(target_inputs, degreeFreedom=args.affine_rot_degree,
                                                          scale=args.affine_scale,
                                                          shearingScale=args.affine_shearing)
                        inputstaug = apply_transform(target_inputs, Theta)
                        outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        outputst_transformed = apply_transform(outputst, Theta)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        pc_loss = alpha * criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    elif args.method == 'A4':
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        outputst, _, _, _, _, _, _, _, _, _ = model(target_inputs)
                        # Need to return a batch of shuffled 2d slices here.
                        batch_trs = target_inputs.cpu().numpy()
                        batch_trs = p.map(
                            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                            np.copy(batch_trs))
                        batch_trs = p.map(
                            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                            np.copy(batch_trs))
                        inputstaug = torch.Tensor(batch_trs).cuda()
                        outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        pc_loss = alpha * criterion(torch.sigmoid(outputst), torch.sigmoid(outputstaug))
                        loss = supervised_loss + pc_loss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    elif args.method == 'A3':
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        outputst, _, _, _, _, _, _, _, _, _ = model(target_inputs)
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
                        outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        outputst_transformed = apply_transform(outputst, Theta)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        outputst_transformed = torch.sigmoid(outputst_transformed)
                        if bool(args.use_fixmatch):
                            # FixMatch thresholding
                            threshold = 1.0 - 1e-6
                            outputst_transformed = outputst_transformed.masked_fill(
                                (outputst_transformed > threshold), 1.0)
                        pc_loss = alpha * criterion(torch.sigmoid(outputstaug), outputst_transformed)
                        loss = supervised_loss + pc_loss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                    elif args.method == 'adversarial':
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        # Training Discriminator
                        model.eval()
                        discriminator.train()

                        # do the sampling here.

                        # Source Domain sampling
                        inputs_source_discriminator = source_inputs
                        # Target batch
                        batch_trs = target_inputs.cpu().numpy()
                        batch_trs = p.map(
                            partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                            np.copy(batch_trs))
                        batch_trs = p.map(
                            partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                            np.copy(batch_trs))
                        inputs_target_discriminator_aug = torch.Tensor(batch_trs).cuda()

                        Theta, Theta_inv = generate_affine(inputs_target_discriminator_aug,
                                                          degreeFreedom=args.affine_rot_degree, scale=args.affine_scale,
                                                          shearingScale=args.affine_shearing)
                        inputs_target_discriminator_aug = apply_transform(inputs_target_discriminator_aug, Theta)

                        inputs_models_discriminator = torch.cat(
                            (inputs_source_discriminator, inputs_target_discriminator_aug), 0)
                        labels_discriminator = to_var_gpu(
                            torch.cat((torch.zeros(inputs_source_discriminator.size(0)),
                                       torch.ones(inputs_target_discriminator_aug.size(0))), 0).type(torch.LongTensor))

                        # print('size Discriminator')
                        # print(inputs_models_discriminator.size())

                        _, _, _, _, _, _, dec4, dec3, dec2, dec1 = model(inputs_models_discriminator)

                        dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
                        dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
                        dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
                        dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')

                        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)

                        discriminator.zero_grad()
                        outputs_discriminator = discriminator(inputs_discriminator)
                        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                                          labels_discriminator)
                        correct += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
                        num_of_subjects += int(outputs_discriminator.size(0))

                        loss_discriminator.backward()
                        optimizer_discriminator.step()
                        discriminator_loss = loss_discriminator.item()
                        # Train model
                        model.train()
                        discriminator.eval()

                        # Here we get a new batch of target domain slices
                        target_batch = next(target_dl)
                        target_inputs, target_labels = (target_batch['inputs'].to(device),
                                                        target_batch['labels'].to(device))
                        outputst, _, _, _, _, _, _, _, _, _ = model(target_inputs)
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

                        model.zero_grad()
                        outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        outputst_transformed = apply_transform(outputst, Theta)

                        inputs_models_discriminator = torch.cat((source_inputs, inputstaug), 0)
                        _, _, _, _, _, _, dec4, dec3, dec2, dec1 = model(inputs_models_discriminator)

                        dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
                        dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
                        dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
                        dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')

                        inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)

                        outputs_discriminator = discriminator(inputs_discriminator)
                        labels_discriminator = to_var_gpu(
                            torch.cat((torch.zeros(source_inputs.size(0)),
                                       torch.ones(inputstaug.size(0))), 0).type(torch.LongTensor))
                        loss_discriminator = torch.nn.CrossEntropyLoss(size_average=True)(outputs_discriminator,
                                                                                          labels_discriminator)
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        pc_loss = alpha * criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
                        adversarial_loss = - beta * loss_discriminator
                        loss = supervised_loss + pc_loss + adversarial_loss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                    elif args.method == 'mean_teacher':
                        model.train()
                        model_t.train()
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
                        model.train()
                        model_t.train()
                        outputs, _, _, _, _, _, _, _, _, _ = model(source_inputs)
                        outputstaug, _, _, _, _, _, _, _, _, _ = model(inputstaug)
                        with torch.no_grad():
                            outputst, _, _, _, _, _, _, _, _, _ = model_t(target_inputs)
                        outputst_transformed = apply_transform(outputst, Theta)
                        pc_loss = criterion(torch.sigmoid(outputstaug), torch.sigmoid(outputst_transformed))
                        supervised_loss = dice_soft_loss(torch.sigmoid(outputs), source_labels)
                        loss = supervised_loss + (alpha) * pc_loss
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        update_ema_variables(model, model_t, 0.9, iteration)
                    #running_loss += loss.item()
                    pbar.update(1)
                    if args.method == 'cyclegan':
                        postfix_dict = {
                                        'acc_discriminator_s': acc_discriminator_s,
                                        'acc_discriminator_t': acc_discriminator_t,
                                        'discriminator_t_loss': loss_discriminator_s.item(),
                                        'discriminator_s_loss': loss_discriminator_t.item(),
                                        'cycle_loss_t': cycle_loss_target.item(),
                                        'cycle_loss_s': cycle_loss_source.item(),
                                       }
                    elif args.method == 'supervised':
                        postfix_dict = {'tumour_loss': tumour_loss.item(),
                                        'cochlea_loss': cochlea_loss.item(),
                                        'supervised_loss': supervised_loss.item()}
                    else:
                        postfix_dict = {'loss': loss.item(),
                                        'pc_loss': pc_loss.item(),
                                        'supervised_loss': supervised_loss.item()}
                    if args.method == 'adversarial':
                        discriminator_accuracy = (correct / num_of_subjects).item()
                        postfix_dict.update({'discriminator_loss': discriminator_loss,
                                             'adversarial_loss': adversarial_loss.item(),
                                             'discriminator_acc': discriminator_accuracy})
                    pbar.set_postfix(postfix_dict)
                    iteration = epoch * len(indxMiniBatch) + indb
                    if iteration % args.tensorboard_every_n == 0:
                        ##### Discriminator validation ######
                        if args.method == 'adversarial':
                            correct_val, num_of_subjects_val = 0, 0
                            for i in range(20):
                                source_val_slice_batch = next(source_val_slice_dl)
                                source_val_slice_inputs = source_val_slice_batch['inputs'].to(device)
                                source_val_slice_labels = source_val_slice_batch['labels'].to(device)
                                target_val_slice_batch = next(target_val_slice_dl)
                                target_val_slice_inputs = target_val_slice_batch['inputs'].to(device)
                                target_val_slice_labels = target_val_slice_batch['labels'].to(device)
                                discriminator.eval()
                                inputs_source_discriminator = source_val_slice_inputs
                                batch_trs = target_val_slice_inputs.cpu().numpy()
                                batch_trs = p.map(
                                    partial(non_geometric_augmentations, method='bias', norm_training_images=None),
                                    np.copy(batch_trs))
                                batch_trs = p.map(
                                    partial(non_geometric_augmentations, method='kspace', norm_training_images=None),
                                    np.copy(batch_trs))
                                inputs_target_discriminator_aug = torch.Tensor(batch_trs).cuda()

                                Theta, Theta_inv = generate_affine(inputs_target_discriminator_aug,
                                                                  degreeFreedom=args.affine_rot_degree, scale=args.affine_scale,
                                                                  shearingScale=args.affine_shearing)
                                inputs_target_discriminator_aug = apply_transform(inputs_target_discriminator_aug, Theta)

                                inputs_models_discriminator = torch.cat(
                                    (inputs_source_discriminator, inputs_target_discriminator_aug), 0)
                                labels_discriminator = to_var_gpu(
                                    torch.cat((torch.zeros(inputs_source_discriminator.size(0)),
                                               torch.ones(inputs_target_discriminator_aug.size(0))), 0).type(torch.LongTensor))
                                _, _, _, _, _, _, dec4, dec3, dec2, dec1 = model(inputs_models_discriminator)
                                dec1 = F.interpolate(dec1, size=dec2.size()[2:], mode='bilinear')
                                dec2 = F.interpolate(dec2, size=dec2.size()[2:], mode='bilinear')
                                dec3 = F.interpolate(dec3, size=dec2.size()[2:], mode='bilinear')
                                dec4 = F.interpolate(dec4, size=dec2.size()[2:], mode='bilinear')
                                inputs_discriminator = torch.cat((dec1, dec2, dec3, dec4), 1)
                                outputs_discriminator = discriminator(inputs_discriminator)
                                correct_val += (torch.argmax(outputs_discriminator, dim=1) == labels_discriminator).float().sum()
                                num_of_subjects_val += int(outputs_discriminator.size(0))
                            discriminator_val_accuracy = (correct_val / num_of_subjects_val).item()
                            writer.add_scalar('discriminator_acc_val', discriminator_val_accuracy, iteration)
                            ##############################################################

                        print(args.tag)
                        #print('Training: [%d, %5d] loss:%.3f ' % (epoch + 1, batch, running_loss / (indb + 1)))
                        performance_target_val_s, \
                        performance_target_val, \
                        performance_target_val_0, \
                        performance_target_val_ema = inference_func(args, p, model, target_val_dataset,
                                                                    prefix=os.path.join(results_folder, 'target_val'),
                                                                    iteration=iteration, infer_on=[0])
                        performance_source_val_s, \
                        performance_source_val, \
                        performance_source_val_0, \
                        performance_source_val_ema = inference_func(args, p, model, source_val_dataset,
                                                                    prefix=os.path.join(results_folder, 'source_val'),
                                                                    iteration=iteration, infer_on=[0])
                        performance_target_train_s, \
                        performance_target_train, \
                        performance_target_train_0, \
                        performance_target_train_ema = inference_func(args, p, model, target_test_dataset,
                                                                      prefix=os.path.join(results_folder,
                                                                                          'target_train'),
                                                                      iteration=iteration, infer_on=[0])  # hack
                        print('*** Target Val Performance ***')
                        print(performance_target_val_s, performance_target_val,
                              performance_target_val_0, performance_target_val_ema)
                        print('*** Source Val Performance ***')
                        print(performance_source_val_s, performance_source_val,
                              performance_source_val_0, performance_source_val_ema)
                        print('*** Target Train Performance ***')
                        print(performance_target_train_s, performance_target_train,
                              performance_target_val_0, performance_target_val_ema)
                        if args.task == 'tumour':
                            for idx, modality in enumerate(['flair', 't1c', 't1', 't2']):
                                save_images(writer=writer, images=source_inputs[:, (idx,), :, :],
                                                           normalize=True, sigmoid=False,
                                                           iteration=iteration, name='source_' + modality)
                                save_images(writer=writer, images=target_inputs[:, (idx,), :, :],
                                                           normalize=True, sigmoid=False,
                                                           iteration=iteration, name='target_' + modality)
                                save_images(writer=writer, images=inputstaug[:, (idx,), :, :],
                                                           normalize=True, sigmoid=False,
                                                           iteration=iteration, name=modality + '_aug')
                        elif args.task == 'ms':
                            save_images(writer=writer, images=source_labels, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='source_labels', png=True)
                            save_images(writer=writer, images=target_labels, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='target_labels', png=True)
                            save_images(writer=writer, images=outputs, normalize=False, sigmoid=True,
                                                   iteration=iteration, name='outputs_source', png=True)
                            save_images(writer=writer, images=source_inputs, normalize=True,
                                                       sigmoid=False, png=True,
                                                       iteration=iteration, name='source_inputs')
                            save_images(writer=writer, images=target_inputs, normalize=True,
                                                       sigmoid=False, png=True,
                                                       iteration=iteration, name='targets_inputs')
                            if inputstaug is not None:
                                save_images(writer=writer, images=inputstaug, normalize=True, sigmoid=False,
                                                           iteration=iteration, name='inputsaug')
                            if outputst is not None:
                                save_images(writer=writer, images=outputst, normalize=False, sigmoid=True,
                                                       iteration=iteration, name='outputs_target')
                        elif args.task == 'crossmoda' and args.method == 'cyclegan':
                            save_images(writer=writer, images=source_inputs, normalize=True,
                                                       sigmoid=False, png=True,
                                                       iteration=iteration, name='source_inputs')
                            save_images(writer=writer, images=target_inputs, normalize=True,
                                                       sigmoid=False, png=True,
                                                       iteration=iteration, name='targets_inputs')
                            save_images(writer=writer, images=cycle_source, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='cycle_source', png=True)
                            save_images(writer=writer, images=cycle_target, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='cycle_target', png=True)
                            save_images(writer=writer, images=generated_source, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='generated_source', png=True)
                            save_images(writer=writer, images=generated_target, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='generated_target', png=True)
                            for key, value in postfix_dict.items():
                                writer.add_scalar('{}/train'.format(key), value, iteration)
                                
                        elif args.task == 'crossmoda' and args.method == 'supervised':
                            save_images(writer=writer, images=source_inputs, normalize=True,
                                                       sigmoid=False, png=False,
                                                       iteration=iteration, name='source_inputs')
                            save_images(writer=writer, images=target_inputs, normalize=True,
                                                       sigmoid=False, png=False,
                                                       iteration=iteration, name='targets_inputs')
                            save_images(writer=writer, images=outputs, normalize=False, sigmoid=True,
                                                   iteration=iteration, name='outputs_source_tumour', png=False)
                            save_images(writer=writer, images=outputs2, normalize=False, sigmoid=True,
                                                   iteration=iteration, name='outputs_source_cochlea', png=False)
                            save_images(writer=writer, images=tumour_labels, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='tumour_labels', png=False)
                            save_images(writer=writer, images=cochlea_labels, normalize=True, sigmoid=False,
                                                   iteration=iteration, name='cochlea_labels', png=False)
                            for key, value in postfix_dict.items():
                                writer.add_scalar('{}/train'.format(key), value, iteration)
                        
                        
                        # save_images(writer=writer, images=outputst_transformed, normalize=False,
                        #                            sigmoid=True,
                        #                            iteration=iteration, name='outputs_target_thresholded')
                        if args.task != 'crossmoda':
                            for key, value in postfix_dict.items():
                                writer.add_scalar('{}/train'.format(key), value, iteration)
                            for key, value in performance_source_val_s.items():
                                writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
                            if performance_source_val is not None:
                                for key, value in performance_source_val.items():
                                    writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
                            for key, value in performance_target_val_s.items():
                                writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
                            if performance_target_val is not None:
                                for key, value in performance_target_val.items():
                                    writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
                            for key, value in performance_target_train_s.items():
                                writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
                            if performance_target_train is not None:
                                for key, value in performance_target_train.items():
                                    writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
                            if performance_source_val_0 is not None:
                                for key, value in performance_source_val_0.items():
                                    writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
                            if performance_source_val_ema is not None:
                                for key, value in performance_source_val_ema.items():
                                    writer.add_scalar('{}/val_source'.format(key), np.mean(value), iteration)
                            if performance_target_val_0 is not None:
                                for key, value in performance_target_val_0.items():
                                    writer.add_scalar('{}/val_target'.format(key), np.mean(value), iteration)
                            if performance_target_val_ema is not None:
                                for key, value in performance_target_val_ema.items():
                                    writer.add_scalar('ema_{}/val_target'.format(key), np.mean(value), iteration)
                            if performance_target_train_0 is not None:
                                for key, value in performance_target_train_0.items():
                                    writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
                            if performance_target_train_ema is not None:
                                for key, value in performance_target_train_ema.items():
                                    writer.add_scalar('{}/train_target'.format(key), np.mean(value), iteration)
                            # writer.add_hparams(vars(args), {'hparam/' + key: value for key, value in postfix_dict.items()})
                            source_inputs_numpy = outputs.cpu().detach().numpy().flatten()
                            source_inputs_numpy = np.random.choice(source_inputs_numpy, size=1000)
                            source_inputs_numpy = 1 / (1 + np.exp(-source_inputs_numpy))  # Manual sigmoid
                            writer.add_histogram('outputs_source_hist', source_inputs_numpy, iteration)
                            if outputst is not None:
                                target_outputs_numpy = outputst.cpu().detach().numpy().flatten()
                                target_outputs_numpy = np.random.choice(target_outputs_numpy, size=1000)
                                target_outputs_numpy = 1 / (1 + np.exp(-target_outputs_numpy))  # Manual sigmoid
                                writer.add_histogram('outputs_target_hist', target_outputs_numpy, iteration)
                    if iteration % args.save_every_n == 0:
                        model.train()
                        print('SAVING MODEL')
                        torch.save(model.state_dict(),
                                   os.path.join(model_folder, run_name + '_{}.pt'.format(iteration)))
                    if iteration >= args.iterations:
                        print('Training ending!')
                        model.train()
                        print('SAVING MODEL')
                        torch.save(model.state_dict(),
                                   os.path.join(model_folder, run_name + '_{}.pt'.format(iteration)))
                        is_training = False
                        break
            end_time = time.time()
            time_epoch = (end_time - start_time) / 60
            print('Time: {}'.format(time_epoch))
    except KeyboardInterrupt:
        print('Interrupted training at iteration {}'.format(iteration))
        model.train()
        print('SAVING MODEL')
        torch.save(model.state_dict(), os.path.join(model_folder, run_name + '_{}.pt'.format(iteration)))
    writer.close()


def main(args):
    def save_images(writer, images, iteration, name, normalize=True, sigmoid=False, k=3,
                    tensorboard=True, png=False):
        if normalize:
            images = (images - images.min()) / (images.max() - images.min())
        if sigmoid:
            images = torch.sigmoid(images)
        images = torch.rot90(images, k=k, dims=[-2, -1])
        grid = torchvision.utils.make_grid(images)
        if tensorboard:
            writer.add_image(name, grid, iteration)
        if png:
            torchvision.utils.save_image(tensor=grid,
                                         fp=os.path.join(writer.log_dir, f"{name}_{iteration}.png"),
                                         format='png'
                                        )
    band = 'new_{}_{}_{}'.format(args.method, args.source, args.target)
    assert args.task in ['ms', 'tumour', 'crossmoda']
    # Directory paths
    model_folder = model_saving_paths[os.uname().nodename]
    results_folder = results_paths[os.uname().nodename]
    tensorboard_folder = tensorboard_paths[os.uname().nodename] 
    seg_optimizer, scheduler_S, criterion, criterion2 = None, None, None, None
    cross_moda_args = None
    discriminator, optimizer_discriminator, U_Teacher = None, None, None
    # Configure Network optimization
    if args.task in ['ms', 'tumour']:
        seg_model = Destilation_student_matchingInstance(args.labels - 1, args.channels)
        seg_model.cuda()
        seg_optimizer = optim.Adam(seg_model.parameters(), lr=args.lr)
        step_1 = 20000 if args.task == 'ms' else 5000
        step_2 = 20000 if args.task == 'ms' else 10000
        scheduler_S = optim.lr_scheduler.MultiStepLR(seg_optimizer, milestones=[step_1, step_2], gamma=0.1)
        criterion = dice_soft_loss if args.loss == 'dice' else bland_altman_loss
        criterion2 = ss_loss
    elif args.task == 'crossmoda' and args.method == 'cyclegan':
        seg_model = SplitHeadModel(args.channels)
        generator_s_t = GeneratorUnet(1, args.channels, use_sigmoid=True)
        generator_t_s = GeneratorUnet(1, args.channels, use_sigmoid=True)
#         seg_model.cuda()
        generator_s_t.cuda()
        generator_t_s.cuda()
        discriminator_s = DiscriminatorCycleGANSimple(1, 2, args.discriminator_complexity)
        discriminator_t = DiscriminatorCycleGANSimple(1, 2, args.discriminator_complexity)
        discriminator_s.cuda()
        discriminator_t.cuda()
        optimizer_discriminator = optim.Adam(
            itertools.chain(discriminator_s.parameters(), discriminator_t.parameters()),
            lr=1e-4)
        optimizer_generator = optim.Adam(
            itertools.chain(generator_s_t.parameters(), generator_t_s.parameters()),
            lr=1e-4)
        step_1 = 20000
        step_2 = 20000
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_generator, milestones=[step_1, step_2], gamma=0.1)
        # Gradient clipping!
        torch.nn.utils.clip_grad_norm(itertools.chain(generator_s_t.parameters(),
                                                      generator_t_s.parameters(),
                                                      discriminator_t.parameters(),
                                                      discriminator_s.parameters())
                                      , 1.0)
        cross_moda_args = (generator_s_t, generator_t_s, discriminator_s,
                           discriminator_t, optimizer_discriminator, optimizer_generator)
    elif args.task == 'crossmoda' and args.method == 'cycada':
        seg_model = SplitHeadModel(args.channels)
        generator_s_t = GeneratorUnet(1, args.channels, use_sigmoid=True)
        generator_t_s = GeneratorUnet(1, args.channels, use_sigmoid=True)
        seg_model.cuda()
        generator_s_t.cuda()
        generator_t_s.cuda()
        discriminator_s = DiscriminatorCycleGANSimple(1, 2, args.discriminator_complexity)
        discriminator_t = DiscriminatorCycleGANSimple(1, 2, args.discriminator_complexity)
        discriminator_s.cuda()
        discriminator_t.cuda()
        optimizer_discriminator = optim.Adam(
            itertools.chain(discriminator_s.parameters(), discriminator_t.parameters()),
            lr=1e-4)
        optimizer_generator = optim.Adam(
            itertools.chain(generator_s_t.parameters(), generator_t_s.parameters()),
            lr=1e-4)
        step_1 = 20000
        step_2 = 20000
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_generator, milestones=[step_1, step_2], gamma=0.1)
        # Gradient clipping!
        torch.nn.utils.clip_grad_norm(itertools.chain(generator_s_t.parameters(),
                                                      generator_t_s.parameters(),
                                                      discriminator_t.parameters(),
                                                      discriminator_s.parameters())
                                      , 1.0)
        cross_moda_args = (generator_s_t, generator_t_s, discriminator_s,
                           discriminator_t, optimizer_discriminator, optimizer_generator)
    elif args.task == 'crossmoda' and args.method == 'supervised':
#         seg_model = get_fpn()
        seg_model = SplitHeadModel(args.channels)
        seg_model.cuda()
        seg_optimizer = optim.Adam(seg_model.parameters(), lr=args.lr)
        scheduler_S = optim.lr_scheduler.MultiStepLR(seg_optimizer, milestones=[20000, 20000], gamma=0.1)
    if args.method == 'adversarial':
        # Discriminator setup #
        discriminator = DiscriminatorDomain(352, 2, args.discriminator_complexity)
        discriminator.cuda()
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4)
        ########################
    elif args.method == 'mean_teacher':
        U_Teacher = Destilation_student_matchingInstance(args.labels - 1, args.channels)
        for param in U_Teacher.parameters():
            param.detach_()
        U_Teacher.cuda()

    starting_epoch = 0
    args.checkpoint = None if args.checkpoint == "null" else args.checkpoint
    if args.checkpoint is not None:
        print('Loading model from checkpoint')
        starting_epoch = int(os.path.basename(args.checkpoint.split('.')[0]).split('_')[-1])
        U_Student.load_state_dict(torch.load(args.checkpoint))

    slice_dataset = {'ms': SliceDataset, 'crossmoda': get_monai_slice_dataset, 'tumour': SliceDatasetTumour}[args.task]
    whole_volume_dataset = {'ms': WholeVolumeDataset, 'crossmoda': WholeVolumeDataset, 'tumour': WholeVolumeDatasetTumour}[args.task]
    source_train_dataset = slice_dataset(ms_path[os.uname().nodename][args.source],
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

    target_train_dataset = slice_dataset(ms_path[os.uname().nodename][args.target],
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
    save_images = partial(save_images, k=1 if args.task == 'tumour' else 3)
    if args.infer:
        p = multiprocessing.Pool(4)
        target_test_dataset = whole_volume_dataset(ms_path[os.uname().nodename][args.target + '_whole'], split='test',
                                                   tumour_only=bool(args.tumour_only), paddtarget=args.paddtarget,
                                                   dataset_split_csv=args.target_split)
        performance_target_train_s, performance_target_train, _, _ = \
            inference_func(args, p, seg_model, target_test_dataset, prefix=os.path.join(results_folder, 'target_test'),
                           epoch=starting_epoch)  # hack
    else:
        train(args=args, model=seg_model, optimizer=seg_optimizer, writer=writer, inference_func=inference_func,
              save_images=save_images, discriminator=discriminator,
              optimizer_discriminator=optimizer_discriminator, model_t=U_Teacher,
              criterion=criterion, criterion2=criterion2, scheduler=scheduler_S,
              run_name='{}_{}'.format(band, args.tag), source_val_slice_dataset=source_val_slice_dataset,
              target_val_slice_dataset=target_val_slice_dataset,
              source_train_dataset=source_train_dataset, target_train_dataset=target_train_dataset,
              source_test_dataset=source_test_dataset, target_test_dataset=target_train_whole_vol_dataset,  # hack
              source_val_dataset=source_val_dataset, target_val_dataset=target_val_dataset,
              model_folder=model_folder, results_folder=results_folder, cross_moda_args=cross_moda_args)

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
