import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import scipy.ndimage.morphology
import numpy as np

"""
# https://github.com/GunhoChoi/3D-Unet-Pytorch/blob/master/3D_UNet_CrossEntropyLoss.ipynb
"""

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def maxpool_3d():
    model = nn.Sequential(
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
    )
    return model

def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        # conv_block_3d(in_dim, out_dim, act_fn),
        # nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        # nn.BatchNorm3d(out_dim),
        conv_block_3d(in_dim, out_dim, act_fn),
        conv_block_3d(out_dim, out_dim, act_fn),
    )
    return model


class MyUnet3d(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        super(MyUnet3d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter

        act_fn = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
        )

        print("\n------Initiating U-Net------\n")
        self.l1_conv = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.l1_seblock = ChannelSELayer3D(self.num_filter)
        self.l1_to_l2 = maxpool_3d()
        self.l2_conv = conv_block_2_3d(self.num_filter * 1, self.num_filter * 2, act_fn)
        self.l2_seblock = ChannelSELayer3D(self.num_filter * 2)
        self.l2_to_l3 = maxpool_3d()
        self.l3_conv = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.l3_seblock = ChannelSELayer3D(self.num_filter * 4)
        self.l3_to_bridge = maxpool_3d()

        self.bridge_conv = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.bridge_to_r3 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.r3_conv = conv_block_2_3d(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.r3_to_r2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.bridge_to_r2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.r2_conv = conv_block_2_3d(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.r2_to_r1 = conv_trans_block_3d(self.num_filter * 2, self.num_filter, act_fn)
        self.r1_conv = conv_block_2_3d(self.num_filter * 2, self.num_filter, act_fn)

        self.r1_fin = nn.Sequential(
            conv_block_2_3d(self.num_filter, self.out_dim, act_fn),
            nn.Softmax(dim=1)
        )

        self.fin = nn.Sequential(
            conv_block_2_3d(self.in_dim+self.out_dim, self.out_dim, act_fn),
            nn.Softmax(dim=1)
        )

    """SEBLOCK"""

    # def forward(self, inp):
    #     out_l1 = self.l1_seblock(self.l1_conv(inp))
    #     out_l2 = self.l2_seblock(self.l2_conv(self.l1_to_l2(out_l1)))
    #     out_l3 = self.l3_seblock(self.l3_conv(self.l2_to_l3(out_l2)))
    #
    #     out_bridge = self.bridge_to_r3(self.bridge_conv(self.l3_to_bridge(out_l3)))
    #
    #     out_r3 = self.r3_conv(torch.cat([out_l3, out_bridge], dim=1))
    #     out_r2 = self.r2_conv(torch.cat([out_l2, self.r3_to_r2(out_r3)], dim=1))
    #     out_r1 = self.r1_conv(torch.cat([out_l1, self.r2_to_r1(out_r2)], dim=1))
    #
    #     out = self.r1_fin(out_r1)
    #     # out = self.fin(torch.cat([inp, self.r1_fin(out_r1)], dim=1))
    #
    #     return out


    """SEBLOCK"""

    def forward(self, inp):
        out_l1 = self.l1_conv(inp)
        out_l2 = self.l2_conv(self.l1_to_l2(out_l1))
        out_l3 = self.l3_conv(self.l2_to_l3(out_l2))

        out_bridge = self.bridge_to_r3(self.bridge_conv(self.l3_to_bridge(out_l3)))

        out_r3 = self.r3_conv(torch.cat([out_l3, out_bridge], dim=1))
        out_r2 = self.r2_conv(torch.cat([out_l2, self.r3_to_r2(out_r3)], dim=1))
        out_r1 = self.r1_conv(torch.cat([out_l1, self.r2_to_r1(out_r2)], dim=1))

        out = self.r1_fin(out_r1)
        # out = self.fin(torch.cat([inp, self.r1_fin(out_r1)], dim=1))

        return out

