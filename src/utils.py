import json
import numpy as np
import torch
import io
import os
import monai
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from rand_bias_field import RandomBiasFieldLayer
from rand_kspace import RandomKSpaceLayer
from rand_hist_norm import RandomHistNormLayer
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import src.medicaldetectiontoolkit.model_utils as mutils
import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def save_images(writer, images, iteration, name, normalize=True, sigmoid=False, k=3,
                    tensorboard=True, png=False):
    if len(images.shape) == 4:
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
    elif len(images.shape) == 5:
        if normalize:
            images = (images - images.min()) / (images.max() - images.min())
        if sigmoid:
            images = torch.sigmoid(images)
        monai.visualize.img2tensorboard.add_animated_gif(writer, name, images[0, ...].cpu().detach().numpy(),
                                                         max_out=images.shape[-1], scale_factor=255,
                                                         global_step=iteration)
        
def create_overlap_image(writer, images, preds, labels, iteration, name, normalize=True, sigmoid=False, k=3,
                    tensorboard=True, png=False):
    if len(images.shape) == 5:
        images = ((images - images.min()) / (images.max() - images.min())).cpu()
        labels = labels.cpu()
        if sigmoid:
            preds = (torch.sigmoid(preds) > 0.5).to(torch.LongTensor()).cpu()
        else:
            preds = (preds > 0.5).to(torch.LongTensor()).cpu()
        true_positives = preds * labels
        # R G B
        true_positives = torch.hstack([torch.zeros_like(true_positives),
                                       true_positives,
                                       torch.zeros_like(true_positives)])
#         true_negatives = (torch.where(preds == 0, 1, 0) *
#                          torch.where(labels == 0, 1, 0))
#         true_negatives = torch.hstack([torch.zeros_like(true_negatives),
#                                        torch.zeros_like(true_negatives),
#                                        true_negatives])
        false_positives = (torch.where(preds == 1, 1, 0) *
                           torch.where(labels == 0, 1, 0))
        false_positives = torch.hstack([
            false_positives,
            torch.zeros_like(false_positives),
            torch.zeros_like(false_positives)])
        false_negatives = (torch.where(preds == 0, 1, 0) *
                           torch.where(labels == 1, 1, 0))
        false_negatives = torch.hstack([
            torch.zeros_like(false_negatives),
            torch.zeros_like(false_negatives),
            false_negatives])
        vid_tensor = images.repeat(1, 3, 1, 1, 1)
        vid_tensor.masked_fill_(true_positives.to(torch.bool), 1)
        vid_tensor.masked_fill_(false_positives.to(torch.bool), 1)
        vid_tensor.masked_fill_(false_negatives.to(torch.bool), 1)
        vid_tensor = vid_tensor.permute(0, 4, 1, 2, 3)
        writer.add_video(name, vid_tensor, global_step=iteration, fps=6)


def soft_dice(seg, target, n_labels):
    ov_dice = np.zeros(n_labels)
    for label_idx in range(0, n_labels):
        n_target = (target == label_idx)
        n_seg = (seg == label_idx)
        intersect = (n_seg * n_target)
        n_target = n_target.astype(np.float)
        n_seg = n_seg.astype(np.float)
        intersect = intersect.astype(np.float)
        eps = 10e-20
        dice = (2 * np.sum(intersect)) / (np.sum(n_target) + np.sum(n_seg) + eps)
        ov_dice[label_idx] = dice
    return ov_dice


def batch_adaptation(x, m, w, synthesis=False):
    # Cropp the slices on x,y axis to avoid too much background.
    if synthesis:
        m = m.unsqueeze(dim=0).unsqueeze(dim=1)
    if x.size(2) > w or x.size(3) > w:
        x_diff = x.size(2) - w
        y_diff = x.size(3) - w
        x_lim = -int(np.ceil(x_diff / 2)) if -int(np.ceil(x_diff / 2)) < 0 else None
        y_lim = -int(np.ceil(y_diff / 2)) if -int(np.ceil(y_diff / 2)) < 0 else None
        x = x[:, :, int(np.floor(x_diff/2)):x_lim, int(np.floor(y_diff/2)):y_lim]
        m = m[:, :, int(np.floor(x_diff/2)):x_lim, int(np.floor(y_diff/2)):y_lim]
    y = torch.zeros(x.size(0), x.size(1), w, w) + x[0, 0, 0, 0]
    n = torch.zeros(x.size(0), 1, w, w)

    siz = [int(np.floor((y.size(2) - x.size(2)) / 2)), int(np.floor((y.size(3) - x.size(3)) / 2))]
    y[:, :, siz[0]:siz[0] + x.size(2), siz[1]:siz[1] + x.size(3)] = x
    n[:, :, siz[0]:siz[0] + x.size(2), siz[1]:siz[1] + x.size(3)] = m
    return y, n


def bland_altman_loss(output, target):
    eps = 1e-19
    l1 = torch.sum(torch.abs(output - target), dim=(2, 3))
    denominator = torch.sum(output, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) + eps
    return 2 * torch.sum(l1 / denominator)


def ss_loss(output, target, alpha):
    s = 1e-19
    s1 = torch.sum(((target - output) * (target - output)) * target) / (torch.sum(target) + s)
    s2 = torch.sum(((target - output) * (target - output)) * (1 - target)) / (torch.sum((1 - target)) + s)
    ss = alpha * s1 + (1 - alpha) * s2
    return ss


def loop_iterable(iterable):
    while True:
        yield from iterable


def apply_transform(inputs, theta):
    grid = F.affine_grid(theta, inputs.size())
    if len(inputs.size()) < 4:
        outputs = F.grid_sample(inputs, grid, mode='nearest', padding_mode="border")
    else:
        outputs = F.grid_sample(inputs, grid, padding_mode="border")
    return outputs


def to_var_gpu(x, cuda_gpu=0):
    if torch.cuda.is_available():
        x = x.cuda(cuda_gpu)
    return Variable(x)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_center_of_mass_slice(label):
    # calculate center of mass of label in through plan direction to select a slice that shows the tumour
    num_slices = label.shape[2]
    slice_masses = np.zeros(num_slices)
    for z in range(num_slices):
        slice_masses[z] = label[:, :, z].sum()

    if sum(slice_masses) == 0:  # if there is no label in the cropped image
        slice_weights = np.ones(num_slices) / num_slices  # give all slices equal weight
    else:
        slice_weights = slice_masses / sum(slice_masses)

    center_of_mass = sum(slice_weights * np.arange(num_slices))
    slice_closest_to_center_of_mass = int(center_of_mass.round())
    return slice_closest_to_center_of_mass


def dice_soft_loss(output, target):
    s = (10e-20)
    intersect = torch.sum(output * target)
    dice = (2 * intersect) / (torch.sum(output) + torch.sum(target) + s)

    return 1 - dice


def generate_affine(inputs, degreeFreedom=10, scale=[0.75, 1.5], shearingScale=[0.5, 0.5], Ngpu=0):
    degree = torch.FloatTensor(inputs.size(0)).uniform_(-degreeFreedom, degreeFreedom) * 3.1416 / 180;
    theta_rotations = torch.zeros(inputs.size(0), 3, 3)

    theta_rotations[:, 0, 0] = torch.cos(degree);
    theta_rotations[:, 0, 1] = torch.sin(degree);
    theta_rotations[:, 1, 0] = -torch.sin(degree);
    theta_rotations[:, 1, 1] = torch.cos(degree);
    theta_rotations[:, 2, 2] = 1

    # degree = torch.cat((torch.FloatTensor(6,1).uniform_(-scale[0],scale[0]), torch.FloatTensor(6,1).uniform_(-scale[1],scale[1])),1)

    degree = torch.FloatTensor(inputs.size(0), 2).uniform_(scale[0], scale[1])

    theta_scale = torch.zeros(inputs.size(0), 3, 3)

    theta_scale[:, 0, 0] = degree[:, 0]
    theta_scale[:, 0, 1] = 0
    theta_scale[:, 1, 0] = 0
    theta_scale[:, 1, 1] = degree[:, 1]
    theta_scale[:, 2, 2] = 1

    degree = torch.cat((torch.FloatTensor(inputs.size(0), 1).uniform_(-shearingScale[0], shearingScale[0]),
                        torch.FloatTensor(inputs.size(0), 1).uniform_(-shearingScale[1], shearingScale[1])), 1)

    theta_shearing = torch.zeros(inputs.size(0), 3, 3)

    theta_shearing[:, 0, 0] = 1
    theta_shearing[:, 0, 1] = degree[:, 0]
    theta_shearing[:, 1, 0] = degree[:, 1]
    theta_shearing[:, 1, 1] = 1
    theta_shearing[:, 2, 2] = 1

    theta = torch.matmul(theta_rotations, theta_scale)
    theta = torch.matmul(theta_shearing, theta)

    theta_inv = torch.inverse(theta)

    theta = to_var_gpu(theta[:, 0:2, :], Ngpu)
    theta_inv = to_var_gpu(theta_inv[:, 0:2, :], Ngpu)

    return theta, theta_inv

def non_geometric_augmentations(inputs, method='kspace', **kwargs):
    if method == 'kspace':
        layer = RandomKSpaceLayer()
        layer.randomise(spatial_rank=2)
    elif method == 'bias':
        layer = RandomBiasFieldLayer()
        layer.randomise(spatial_rank=2)
    elif method == 'histnorm':
        layer = RandomHistNormLayer()
        if not layer.is_ready() and 'norm_training_images' in kwargs:
            layer.train(kwargs['norm_training_images'])
    else:
        raise NotImplementedError
    return layer.layer_op(inputs, None)

def load_default_config(task, dims):
    
    config_dict = dict(
        ms=json.load(open('config/default_ms_config.json', 'r')),
        ms_3d=json.load(open('config/default_ms_3d_config.json', 'r')),
        tumour=json.load(open('config/default_tumour_config.json', 'r')),
        crossmoda=json.load(open('config/default_crossmoda_config.json', 'r')),
        crossmoda_3d=json.load(open('config/default_crossmoda_3d_config.json', 'r')),
        microbleed=json.load(open('config/default_microbleed_config.json', 'r')),
        microbleed_3d=json.load(open('config/default_microbleed_3d_config.json', 'r')),
    )
    return config_dict[task + '_3d' if dims==3 else task]

def save_bboxes_for_plotting(writer, img, results_dict, name, iteration):
    """
    Plot bboxes over first image in batch
    'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                      {'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]}
    """
    boxes = results_dict['boxes']
    box_results_list = results_dict['box_results_list']
    axis_shape = int(np.ceil(img.shape[0]**0.5))
    
    fig, axes = plt.subplots(axis_shape, axis_shape, gridspec_kw = {'wspace':0, 'hspace':0})
    for image_idx, ax in enumerate(fig.axes):
        if image_idx >= img.shape[0]:
            break
        ax.imshow(img.cpu().numpy()[image_idx, 0, ...], cmap='Greys_r', origin='lower', aspect='auto')
        for box_idx, box in enumerate(boxes[image_idx]):
            coords = box['box_coords']
            box_score = box.get('box_score', np.nan)
            box_type = box.get('box_type', np.nan)
            box_pred_class_id = box.get('box_pred_class_id', np.nan)
            #print(coords, box_score, box_type, box_pred_class_id)
            # Given coords draw a square
            rect = Rectangle((coords[0], coords[1]),
                             coords[2]-coords[0],
                             coords[3]-coords[1],
                             linewidth=1,
                             edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        for gt_box_idx, box in enumerate(box_results_list[image_idx]):
            if box.get('box_type', False) == 'gt':
                # Add the patch to the Axes
                gt_coords = box['box_coords']
                # Given coords draw a square
                rect = Rectangle((gt_coords[0], gt_coords[1]),
                                 gt_coords[2]-gt_coords[0],
                                 gt_coords[3]-gt_coords[1],
                                 linewidth=1,
                                 edgecolor='g', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
        ax.axis('off')
    writer.add_figure(tag=name, figure=fig, global_step=iteration)
    return fig

def create_bbox_overlap_volume(writer,
                               images, results_dict,
                               iteration, name, normalize=True, sigmoid=False, k=3,
                               tensorboard=True, png=False):
    # images.shape B, C, H, W, D
    boxes = results_dict['boxes']
    box_results_list = results_dict['box_results_list']
    bboxes_data = []
    images = (images - images.min()) / (images.max() - images.min())
    gt_outlines, gt_mask_outlines = [], []
    pred_outlines, pred_mask_outlines = [], []
    for batch_ix in range(images.shape[0]):
        main_pred_mask_outline = np.zeros_like(images[0].cpu().numpy()[0])
        main_gt_mask_outline = np.zeros_like(images[0].cpu().numpy()[0])
#         gt_mask_outline = torch.tensor(main_pred_mask_outline).repeat(3, 1, 1, 1)
#         gt_outline = torch.tensor(main_pred_mask_outline).repeat(3, 1, 1, 1)
        gt_bboxes = [box['box_coords'] for box in box_results_list[batch_ix] if box.get('box_type', False) == 'gt']
        pred_bboxes = [(box['box_coords']) for box in boxes[batch_ix] if box.get('box_type', False) == 'det']
        if gt_bboxes and pred_bboxes:
            ious, _ = mutils.bbox_overlaps_3D(torch.tensor(gt_bboxes),
                                              torch.tensor(pred_bboxes)).max(dim=0)
            # y1, x1, y2, x2, z1, z2
            def plot_bbox(bbox, initial_mask=None, iou=1, cmap='RdYlGn', return_pytorch_mask=True):
                # Cube is composed of a series of lines...
                # Outline should be difference between 2 cubes, with 1 voxel difference
                # Fill in outer cube, subtract inner cube
                if initial_mask == None:
                    mask = np.zeros_like(images[batch_ix].cpu().numpy()[0])
                else:
                    mask = initial_mask.copy()
                y1, x1, y2, x2, z1, z2 = bbox
                x1, y1, x2, y2, z1, z2 = int(x1), int(y1), int(x2), int(y2), int(z1), int(z2)
                mask[x1:x2, y1:y2, z1:z2] = iou
                mask[x1+1:x2-1, y1+1:y2-1, z1:z2] = 0

                output_array = plt.get_cmap(cmap)(mask)[..., :3]
                output_array = np.transpose(output_array, axes=(3, 0, 1, 2))
                if return_pytorch_mask:
                    return torch.tensor(output_array), torch.tensor(mask).repeat(3, 1, 1, 1)
                else:
                    return torch.tensor(output_array), mask
#             gt_outline, gt_mask_outline = plot_bbox(gt_bboxes[0])
            for gt_bbox in gt_bboxes:
                _, gt_mask_outline = plot_bbox(gt_bbox, return_pytorch_mask=False)
                main_gt_mask_outline = np.logical_or(main_gt_mask_outline, gt_mask_outline)
            # Need to output a combined mask and a combined coloured mask
            for pred_bbox, iou in zip(pred_bboxes, ious):
                _, pred_mask_outline = plot_bbox(pred_bbox, cmap='bwr', return_pytorch_mask=False, iou=1)
                main_pred_mask_outline = np.logical_or(main_pred_mask_outline, pred_mask_outline)
        main_pred_outline = plt.get_cmap('bwr')(main_pred_mask_outline.astype(float))[..., :3]
        main_pred_outline = torch.tensor(np.transpose(main_pred_outline, axes=(3, 0, 1, 2)))
        main_gt_outline = plt.get_cmap('RdYlGn')(main_gt_mask_outline.astype(float))[..., :3]
        main_gt_outline = torch.tensor(np.transpose(main_gt_outline, axes=(3, 0, 1, 2)))
        gt_outlines.append(main_gt_outline)
        gt_mask_outlines.append(torch.tensor(main_gt_mask_outline).repeat(3, 1, 1, 1))
        pred_outlines.append(main_pred_outline)
        pred_mask_outlines.append(torch.tensor(main_pred_mask_outline).repeat(3, 1, 1, 1))
        
    gt_outlines = torch.stack(gt_outlines)
    gt_mask_outlines = torch.stack(gt_mask_outlines)
    pred_outlines = torch.stack(pred_outlines)
    pred_mask_outlines = torch.stack(pred_mask_outlines)
    
    
    vid_tensor = images.repeat(1, 3, 1, 1, 1).detach().cpu()
    
    # Adding the pred
    vid_tensor.masked_fill_(pred_mask_outlines.to(torch.bool), 0)
    pred_outlines.masked_fill_(~pred_mask_outlines.to(torch.bool), 0)
    vid_tensor = vid_tensor + pred_outlines
    
    # Adding the gt
    vid_tensor.masked_fill_(gt_mask_outlines.to(torch.bool), 0)
    gt_outlines.masked_fill_(~gt_mask_outlines.to(torch.bool), 0)
    vid_tensor = vid_tensor + gt_outlines
    
    
    vid_tensor = vid_tensor.permute(0, 4, 1, 2, 3)
    writer.add_video(name, vid_tensor, global_step=iteration, fps=6)


def apply_affine_to_coords(coords, affine):
    """
    This function expects the coords to be in medicaldetectiontoolkit format, i.e
    (y1, x1, y2, x2, z1, z2)

    expects affine to be a numpy array
    """
    # convert from medicaldetectiontoolkit to a 2d array of x, y, (z) coords
    if coords.size == 4: # 2D case
        standard_coords = coords[np.array([1, 0, 3, 2])].reshape(2, 2).copy()
        # need to add a column of 1s to make it a 3-vector of coordinates
        standard_coords = np.hstack([standard_coords, np.ones((2, 1))])
        new_coords = np.matmul(affine, standard_coords.T)
        # convert back to medicaldetectiontoolkit format
        new_coords = new_coords[:2].ravel()[np.array([2, 0, 3, 1])]

    elif coords.size == 6: # 3D case
        standard_coords = coords[np.array([1, 0, 4, 3, 2, 5])].reshape(2, 3).copy()
        # need to add a column of 1s to make it a 4-vector of coordinates
        standard_coords = np.hstack([standard_coords, np.ones((2, 1))])
        new_coords = np.matmul(affine, standard_coords.T)
        new_coords = new_coords[:3].ravel()[np.array([2, 0, 3, 1, 4, 5])]
    else:
        raise Exception('Incorrect bbox format, must be of size 4 or 6, size found: {}'.format(coords.size))

    return new_coords.astype(int)

def collate_patches_object_detection(patch_detections):
    """
    This function expects a list of tuples of
    (
        patch_coord: np.array in [[x0, x1], [y0, y1], ([z0, z1])] format,
        detections: np.array in y0, x0, y1, x1, (z0, z1),
        box_score: scalar
    )
    returns: np.array
    """
    # Off-setting by the initial coordinate
    detections, scores = [], []
    for grid_coord, detection, score in patch_detections:
        # x coordinate
        detection[1] += grid_coord[0][0]
        detection[3] += grid_coord[0][0]
        # y coordinate
        detection[0] += grid_coord[1][0]
        detection[2] += grid_coord[1][0]
        if len(grid_coord) == 3:
            # z coordinate
            detection[4] += grid_coord[2][0]
            detection[5] += grid_coord[2][0]
        detections.append(detection)
        scores.append(score)
    return np.array(detections), np.array(scores)
    
