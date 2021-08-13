import json
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from rand_bias_field import RandomBiasFieldLayer
from rand_kspace import RandomKSpaceLayer
from rand_hist_norm import RandomHistNormLayer
import torchvision

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

def load_default_config(task):
    assert task in ['ms', 'tumour', 'crossmoda']
    config_dict = dict(
        ms=json.load(open('config/default_ms_config.json', 'r')),
        tumour=json.load(open('config/default_tumour_config.json', 'r')),
        crossmoda=json.load(open('config/default_crossmoda_config.json', 'r'))
    )
    return config_dict[task]
