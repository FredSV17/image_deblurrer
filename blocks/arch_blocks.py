import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
import torch.nn as nn
import torchvision.models as models


import torch
import torch.nn as nn

from blocks.base_blocks import BaseBlocks, UnetSkipConnectionBlock
import functools

class UnetGenerator(nn.Module):
    """Create a Unet-based generator (CycleGANs paper architecture)"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)






class Discriminator(BaseBlocks):
    def __init__(self, in_channels=3, features=64, norm_layer=nn.BatchNorm2d, with_dropout=True):
        super(Discriminator, self).__init__()
        self.norm_layer = norm_layer

        self.conv1 = self.conv_block(in_channels, features)            
        self.drop1 = nn.Dropout(0.2) if with_dropout else nn.Identity()

        self.conv2 = self.conv_block(features, features * 2)           
        self.drop2 = nn.Dropout(0.4) if with_dropout else nn.Identity()

        self.conv3 = nn.Conv2d(features * 2, features * 4,
                               kernel_size=4, stride=2, padding=1)   
        self.drop3 = nn.Dropout(0.6) if with_dropout else nn.Identity()

        # Output 1-channel patch prediction
        self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=4, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.drop1(x)

        x = self.leaky_relu(self.conv2(x))
        x = self.drop2(x)

        x = self.leaky_relu(self.conv3(x))
        x = self.drop3(x)

        return self.conv4(x)   # [B,1,H/8,W/8]
    

class LightDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(LightDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=4, padding=0)   # 640 → 160
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=4, padding=0)   # 160 → 40
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=4, padding=0)  # 40 → 10
        self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=2, stride=1, padding=0)         # 10 → 1

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  # [B, 64, 160, 160]
        x = self.leaky_relu(self.conv2(x))  # [B, 128, 40, 40]
        x = self.leaky_relu(self.conv3(x))  # [B, 256, 10, 10]
        x = self.conv4(x)                   # [B, 1, 1, 1]
        return x

# PatchGAN discriminator - good for focusing on finer details of the image
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    