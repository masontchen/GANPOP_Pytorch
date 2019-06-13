import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.spectral_norm import spectral_norm


def conv_block(in_dim,out_dim,act_fn,spec_norm=True):
    if spec_norm:
        model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1)),
            # nn.BatchNorm2d(out_dim),
            act_fn,
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )
    return model


def conv_trans_block(in_dim,out_dim,act_fn,spec_norm=True):
    if spec_norm:
        model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1)),
            # nn.BatchNorm2d(out_dim),
            act_fn,
        )
    else:
        model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn,spec_norm=True):
    if spec_norm:
        model = nn.Sequential(
            conv_block(in_dim,out_dim,act_fn,spec_norm),
            conv_block(out_dim,out_dim,act_fn,spec_norm),
            spectral_norm(nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1)),
            # nn.BatchNorm2d(out_dim),
        )
    else:
        model = nn.Sequential(
            conv_block(in_dim, out_dim, act_fn, spec_norm=False),
            conv_block(out_dim, out_dim, act_fn, spec_norm=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )
    return model
