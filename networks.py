import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('medicaldetectiontoolkit')
from fpn import FPN
from model_utils import NDConvGenerator
from dataclasses import dataclass

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