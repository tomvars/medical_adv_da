import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging
from monai.networks.nets import BasicUNet
from src.medicaldetectiontoolkit.fpn import FPN
from src.medicaldetectiontoolkit.model_utils import NDConvGenerator
from src.medicaldetectiontoolkit.retina_unet import net as retina_unet
from dataclasses import dataclass, field
import numpy as np

logging.basicConfig(level = logging.INFO)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlockInstance(nn.Module):
    def __init__(self, in_channels, out_channels,k ,padd , dropout=False):
        super(_EncoderBlockInstance, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padd),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=padd),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlockInstance(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlockInstance, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3,padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(middle_channels)
        self.leakyrl = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1)
        self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.leakyrl(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.leakyrl(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return x


class Destilation_student_matchingInstance(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(Destilation_student_matchingInstance, self).__init__()

        self.enc1 = _EncoderBlockInstance(num_channels, 64, 5, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _EncoderBlockInstance(64, 96, 3, 1)
        self.enc3 = _EncoderBlockInstance(96, 128, 3, 1)
        self.enc4 = _EncoderBlockInstance(128, 256, 3, 1, dropout=True)
        self.center = _DecoderBlockInstance(256, 512, 256)
        self.dec4 = _DecoderBlockInstance(512, 256, 128)
        self.dec3 = _DecoderBlockInstance(256, 128, 96)
        self.dec2 = _DecoderBlockInstance(96 * 2, 96, 64)

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return (final, enc1, enc2, enc3, enc4, center, dec4, dec3, dec2, dec1)
    
class GeneratorUnet(nn.Module):
    def __init__(self, num_classes, num_channels, use_sigmoid=False):
        super(GeneratorUnet, self).__init__()

        self.enc1 = _EncoderBlockInstance(num_channels, 64, 5, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _EncoderBlockInstance(64, 96, 3, 1)
        self.enc3 = _EncoderBlockInstance(96, 128, 3, 1)
        self.enc4 = _EncoderBlockInstance(128, 256, 3, 1, dropout=True)
        self.center = _DecoderBlockInstance(256, 512, 256)
        self.dec4 = _DecoderBlockInstance(512, 256, 128)
        self.dec3 = _DecoderBlockInstance(256, 128, 96)
        self.dec2 = _DecoderBlockInstance(96 * 2, 96, 64)

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        if self.use_sigmoid:
            final = self.sigmoid(final)
        return final

class SplitHeadModel(nn.Module):
    def __init__(self, num_channels):
        super(SplitHeadModel, self).__init__()

        self.enc1 = _EncoderBlockInstance(num_channels, 64, 5, 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _EncoderBlockInstance(64, 96, 3, 1)
        self.enc3 = _EncoderBlockInstance(96, 128, 3, 1)
        self.enc4 = _EncoderBlockInstance(128, 256, 3, 1, dropout=True)
        self.center = _DecoderBlockInstance(256, 512, 256)
        self.dec4 = _DecoderBlockInstance(512, 256, 128)
        self.dec3 = _DecoderBlockInstance(256, 128, 96)
        self.dec2 = _DecoderBlockInstance(96 * 2, 96, 64)

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.final2 = nn.Conv2d(64, 1, kernel_size=1)

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        final2 = self.final2(dec1)
        return (final, final2, enc1, enc2, enc3, enc4, center, dec4, dec3, dec2, dec1)

class DiscriminatorDomain(nn.Module):
    def __init__(self, num_channels, num_classes, complexity):
        super(DiscriminatorDomain, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, int(8 * complexity), kernel_size=3, stride=2)
        self.BN1 = nn.BatchNorm2d(int(8 * complexity))
        self.conv2 = nn.Conv2d(int(8 * complexity), int(16 * complexity), kernel_size=3, stride=2)
        self.BN2 = nn.BatchNorm2d(int(16 * complexity))
        self.conv3 = nn.Conv2d(int(16 * complexity), int(32 * complexity), kernel_size=3, stride=2)
        self.BN3 = nn.BatchNorm2d(int(32 * complexity))
        self.conv4 = nn.Conv2d(int(32 * complexity), int(64 * complexity), kernel_size=3, stride=2)
        self.BN4 = nn.BatchNorm2d(int(64 * complexity))
        self.fc1 = nn.Linear(int(64 * 15 * 15 * complexity), int(128 * complexity))
        self.drop_1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(128 * complexity), int(64 * complexity))
        self.drop_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(int(64 * complexity), num_classes)
        # self.fc4 = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.BN1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.BN4(self.conv4(x)), 0.2)

        complexity = x.size(1)
        x = x.view(-1, int(x.size(2) * x.size(3) * complexity))
        x = F.relu(self.drop_1(self.fc1(x)))
        x = F.relu(self.drop_2(self.fc2(x)))
        x = self.fc3(x)

        return x
    
class DiscriminatorCycleGAN(nn.Module):
    def __init__(self, num_channels, num_classes, complexity):
        super(DiscriminatorCycleGAN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, int(8 * complexity), kernel_size=3, stride=2)
        self.BN1 = nn.InstanceNorm2d(int(8 * complexity))
        self.conv2 = nn.Conv2d(int(8 * complexity), int(16 * complexity), kernel_size=3, stride=2)
        self.BN2 = nn.InstanceNorm2d(int(16 * complexity))
        self.conv3 = nn.Conv2d(int(16 * complexity), int(32 * complexity), kernel_size=3, stride=2)
        self.BN3 = nn.InstanceNorm2d(int(32 * complexity))
        self.conv4 = nn.Conv2d(int(32 * complexity), int(64 * complexity), kernel_size=3, stride=2)
        self.BN4 = nn.InstanceNorm2d(int(64 * complexity))
        self.fc1 = nn.Linear(int(64 * 15 * 15 * complexity), int(128 * complexity))
        self.drop_1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(int(128 * complexity), int(64 * complexity))
        self.drop_2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(int(64 * complexity), num_classes)
        # self.fc4 = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.BN1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.BN4(self.conv4(x)), 0.2)

        complexity = x.size(1)
        x = x.view(-1, int(x.size(2) * x.size(3) * complexity))
        x = F.relu(self.drop_1(self.fc1(x)))
        x = F.relu(self.drop_2(self.fc2(x)))
        x = self.fc3(x)

        return x
    
class DiscriminatorCycleGANSimple(nn.Module):
    def __init__(self, num_channels, num_classes, complexity):
        super(DiscriminatorCycleGANSimple, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, int(8 * complexity), kernel_size=3, stride=2)
        self.BN1 = nn.InstanceNorm2d(int(8 * complexity))
        self.conv2 = nn.Conv2d(int(8 * complexity), int(16 * complexity), kernel_size=3, stride=2)
        self.BN2 = nn.InstanceNorm2d(int(16 * complexity))
        self.conv3 = nn.Conv2d(int(16 * complexity), int(32 * complexity), kernel_size=3, stride=2)
        self.BN3 = nn.InstanceNorm2d(int(32 * complexity))
        self.conv4 = nn.Conv2d(int(32 * complexity), int(64 * complexity), kernel_size=3, stride=2)
        self.BN4 = nn.InstanceNorm2d(int(64 * complexity))
        self.conv5 = nn.Conv2d(int(64 * complexity), 1, kernel_size=3, stride=2)
        # self.fc4 = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.BN1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.BN4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
#         complexity = x.size(1)
#         x = x.view(-1, int(x.size(2) * x.size(3) * complexity))
#         x = F.relu(self.drop_1(self.fc1(x)))
#         x = F.relu(self.drop_2(self.fc2(x)))
#         x = self.fc3(x)

        return x

def get_fpn():
    @dataclass
    class Config:
        start_filts: int = 48
        end_filts: int = 48*4  # start_filts * 4
        res_architecture: str = 'resnet50'
        sixth_pooling: bool = True
        n_channels: int = 1
        n_latent_dims: int = 0
        num_seg_classes: int = 3
        norm: str = 'instance_norm'
        relu: str = 'leaky_relu'
    cf = Config()
    conv = NDConvGenerator(2)
    model = FPN(cf=cf, conv=conv, operate_stride1=True)
    return model

def get_2d_retina_unet():
    logger = logging.getLogger()
    
    @dataclass
    class Config:
        head_classes: int = 2
        start_filts: int = 48
        end_filts: int = 48*4  # start_filts * 4
        res_architecture: str = 'resnet50'
        sixth_pooling: bool = False
        n_channels: int = 1
        n_latent_dims: int = 0
        num_seg_classes: int = 2
        norm: str = 'instance_norm'
        relu: str = 'leaky_relu'
        n_rpn_features: int = 512 # 128 in 3D
        n_rpn_ancho_ratios: list = field(default_factory=lambda: [0.5, 1, 2])
        rpn_train_anchors_per_image: int = 6
        anchor_matching_iou: float = 0.5
        n_anchors_per_pos: int = 3 # len(self.rpn_anchor_ratios) * 3
        rpn_anchor_stride: int = 1
        pre_nms_limit: int = 3000 #3000 if self.dim == 2 else 6000
        rpn_bbox_std_dev: np.array = field(default_factory=lambda: np.array([0.1, 0.1, 0.2, 0.2]))
        dim: int = 2
        scale: np.array = field(default_factory=lambda: np.array([256, 256, 256, 256]))#np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1]])
        window: np.array = field(default_factory=lambda: np.array([0, 0, 256, 256]))
        detection_nms_threshold: float = 1e-5
        model_max_instances_per_batch_element: int = 10 #10 if self.dim == 2 else 30
        model_min_confidence: float = 0.1
        weight_init: str = None
        patch_size: np.array = field(default_factory=lambda: np.array([256, 256]))
        backbone_path: str = '/home/tom/DomainAdaptationJournal/src/medicaldetectiontoolkit/fpn.py'
        operate_stride1: int = True
        pyramid_levels: list = field(default_factory=lambda: [0, 1, 2, 3])
        rpn_anchor_scales: dict = field(default_factory=lambda: {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]})
        rpn_anchor_ratios: list = field(default_factory=lambda: [0.5, 1, 2])
        backbone_strides: dict = field(default_factory=lambda: {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]})
    cf = Config()
    cf.backbone_shapes = np.array(
                    [[int(np.ceil(cf.patch_size[0] / stride)),
                      int(np.ceil(cf.patch_size[1] / stride))]
                    for stride in cf.backbone_strides['xy']])
    return retina_unet(cf=cf, logger=logger)

def get_3d_retina_unet(spatial_size,
                       base_rpn_anchor_scale_xy=2,
                       base_rpn_anchor_scale_z=2,
                       base_backbone_strides_xy=4,
                       base_backbone_strides_z=1):
    logger = logging.getLogger()
    
    @dataclass
    class Config:
        head_classes: int = 2
        start_filts: int = 48
        end_filts: int = 48*4  # start_filts * 4
        res_architecture: str = 'resnet101'
        sixth_pooling: bool = True
        n_channels: int = 1
        n_latent_dims: int = 0
        num_seg_classes: int = 2
        norm: str = 'instance_norm'
        relu: str = 'leaky_relu'
        n_rpn_features: int = 128 # 128 in 3D
        rpn_anchor_ratios: list = field(default_factory=lambda: [0.5, 1, 2])
        rpn_train_anchors_per_image: int = 30
        anchor_matching_iou: float = 0.3
        roi_chunk_size: int = 600
        n_anchors_per_pos: int = 9 #len(cf.rpn_anchor_ratios) * 3
        rpn_anchor_stride: int = 1
        pre_nms_limit: int = 6000 #3000 if self.dim == 2 else 6000
        rpn_bbox_std_dev: np.array = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        bbox_std_dev: np.array = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        dim: int = 3
        scale: np.array = field(default_factory=lambda: np.array([spatial_size[0], spatial_size[1],
                                                                  spatial_size[0], spatial_size[1],
                                                                  spatial_size[2], spatial_size[2]]))
        window: np.array = field(default_factory=lambda: np.array([0, 0, spatial_size[0],
                                                                   spatial_size[1], 0,
                                                                   spatial_size[2]]))
        detection_nms_threshold: float = 0.5 # Originally 1e-5 Consider changing to 0.1
        model_max_instances_per_batch_element: int = 30 #10 if self.dim == 2 else 30
        model_min_confidence: float = 0.4
        weight_init: str = None
        patch_size: np.array = field(default_factory=lambda: np.array(spatial_size))
        backbone_path: str = '/home/tom/DomainAdaptationJournal/src/medicaldetectiontoolkit/fpn.py'
        operate_stride1: int = True
        pyramid_levels: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
        rpn_anchor_scales: dict = field(default_factory=lambda: {'xy': [[base_rpn_anchor_scale_xy],
                                                                        [base_rpn_anchor_scale_xy*2],
                                                                        [base_rpn_anchor_scale_xy*4],
                                                                        [base_rpn_anchor_scale_xy*8],
                                                                        [base_rpn_anchor_scale_xy*16]
                                                                       ],
                                                                 'z': [[base_rpn_anchor_scale_z],
                                                                       [base_rpn_anchor_scale_z*2],
                                                                       [base_rpn_anchor_scale_z*4],
                                                                       [base_rpn_anchor_scale_z*8],
                                                                       [base_rpn_anchor_scale_z*16]
                                                                      ]})
        backbone_strides: dict = field(default_factory=lambda:  {'xy': [base_backbone_strides_xy,
                                                                        base_backbone_strides_xy*2,
                                                                        base_backbone_strides_xy*4,
                                                                        base_backbone_strides_xy*8,
                                                                        base_backbone_strides_xy*16,
                                                                       ],
                                                                 'z': [base_backbone_strides_z,
                                                                       base_backbone_strides_z*2,
                                                                       base_backbone_strides_z*4,
                                                                       base_backbone_strides_z*8,
                                                                       base_backbone_strides_z*16,
                                                                      ]})
    cf = Config()
    cf.backbone_shapes = np.array(
                [[int(np.ceil(cf.patch_size[0] / stride)),
                  int(np.ceil(cf.patch_size[1] / stride)),
                  int(np.ceil(cf.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(cf.backbone_strides['xy'], cf.backbone_strides['z']
                                             )])
    cf.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            cf.rpn_anchor_scales['xy']]
    cf.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                   cf.rpn_anchor_scales['z']]
    cf.operate_stride1 = True
    return retina_unet(cf=cf, logger=logger)

class BasicUNetFeatures(BasicUNet):
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return (logits, x1, x2, x3, x4, u4, u3, u2, u1)
    
class Discriminator3D(nn.Module):
    def __init__(self, num_channels, num_classes, complexity):
        super(Discriminator3D, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, int(8 * complexity), kernel_size=3, stride=2)
        self.BN1 = nn.InstanceNorm3d(int(8 * complexity))
        self.conv2 = nn.Conv3d(int(8 * complexity), int(16 * complexity), kernel_size=3, stride=2)
        self.BN2 = nn.InstanceNorm3d(int(16 * complexity))
        self.conv3 = nn.Conv3d(int(16 * complexity), int(32 * complexity), kernel_size=3, stride=2)
        self.BN3 = nn.InstanceNorm3d(int(32 * complexity))
        self.conv4 = nn.Conv3d(int(32 * complexity), int(64 * complexity), kernel_size=(3, 3, 1), stride=2)
        self.BN4 = nn.InstanceNorm3d(int(64 * complexity))
        self.conv5 = nn.Conv3d(int(64 * complexity), 1, kernel_size=(3, 3, 1), stride=2)
        self.fc1 = nn.Linear(9, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.BN1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.BN2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.BN3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.BN4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        complexity = x.size(1)
        x = x.view(-1, int(x.size(2) * x.size(3) * complexity))
        x = self.fc1(x)

        return x
     