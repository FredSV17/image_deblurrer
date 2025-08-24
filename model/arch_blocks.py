import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
import torch.nn as nn
import torchvision.models as models


import torch
import torch.nn as nn

from model.base_blocks import BaseBlocks, UnetSkipConnectionBlock
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



        
class DeepUNetBlock(BaseBlocks):
    def __init__(self, in_channels=3, out_channels=3, features=32, norm_layer=nn.BatchNorm2d):
        super(DeepUNetBlock, self).__init__()
        self.norm_layer = norm_layer
        # Encoder
        self.down1 = self.conv_block(in_channels, features)
        self.down2 = self.conv_block(features, features * 2)
        self.down3 = self.conv_block(features * 2, features * 4)
        self.down4 = self.conv_block(features * 4, features * 4)

        # Bottleneck
        self.pre_bottleneck = self.conv_block(features * 4, features * 4)
        self.bottleneck1 = self.bottleneck_block(features * 4, features * 8)
        self.bottleneck2 = self.bottleneck_block(features * 8, features * 16)
        self.bottleneck3 = self.bottleneck_block(features * 16, features * 8)
        self.bottleneck4 = self.bottleneck_block(features * 8, features * 4)
        
        # Decoder
        self.up1 = self.conv_upsample_block(features * 4, features * 4)          # 80 -> 160
        self.up2 = self.conv_upsample_block(features * 8, features * 4)          # 160 -> 320
        self.up3 = self.conv_upsample_block(features * 8, features * 2)
        self.up4 = self.conv_upsample_block(features * 4, features)
        
        # Final output
        self.final = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)  
        d2 = self.down2(d1) 
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Bottleneck
        pre_bottleneck = self.pre_bottleneck(d4)
        bottleneck1 = self.bottleneck1(pre_bottleneck)
        bottleneck2 = self.bottleneck2(bottleneck1)
        bottleneck3 = self.bottleneck3(bottleneck2)
        bottleneck4 = self.bottleneck4(bottleneck3)

        # Decoder with skip connections
        up1 = self.up1(bottleneck4)        
        up1 = torch.cat([up1, d4], dim=1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, d3], dim=1)
        
        up3 = self.up3(up2)
        up3 = torch.cat([up3, d2], dim=1)
        
        up4 = self.up4(up3)
        up4 = torch.cat([up4, d1], dim=1)        
        
        final = self.final(up4)
        out = self.tanh(final)
        return out






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
    
# class LightDiscriminator(nn.Module):
#     def __init__(self, in_channels=3, features=64):
#         super(LightDiscriminator, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)   # 128× 128× 64
#         self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)   # 64× 64× 128
#         self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)  # 32× 32× 256
#         self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=4, stride=2, padding=1) # 16× 16x 1    

#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         x = self.leaky_relu(self.conv1(x))  
#         x = self.leaky_relu(self.conv2(x))  
#         x = self.leaky_relu(self.conv3(x))  
#         x = self.conv4(x)                   
#         return x
    

class LightUNetBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32):
        super(LightUNetBlock, self).__init__()

        # Encoder
        self.down1 = self.conv_block(in_channels, features)          # 640 -> 320
        self.down2 = self.conv_block(features, features * 2)         # 320 -> 160

        # Bottleneck
        self.bottleneck = self.conv_block(features * 2, features * 4)  # 160 -> 80

        # Decoder
        self.up1 = self.conv_upsample_block(features * 4, features * 2)          # 80 -> 160
        self.up2 = self.conv_upsample_block(features * 2 * 2, features)          # 160 -> 320

        # Final output
        self.final = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)  # 640 -> 320
        d2 = self.down2(d1) # 320 -> 160

        # Bottleneck
        bottleneck = self.bottleneck(d2)  # 160 -> 80

        # Decoder with skip connections
        up1 = self.up1(bottleneck)        # 80 -> 160
        up1 = torch.cat([up1, d2], dim=1)

        up2 = self.up2(up1)               # 160 -> 320
        up2 = torch.cat([up2, d1], dim=1)

        out = self.final(up2)             # 320 -> 640
        out = self.tanh(out)
        return out
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(ConvBlock, self).__init__()

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
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels=3, features=64):
#         super(ConvBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=4, padding=0)   # 640 → 160
#         self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=4, padding=0)   # 160 → 40
#         self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=4, padding=0)  # 40 → 10
#         self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=10, stride=1, padding=0)         # 10 → 1

#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         x = self.leaky_relu(self.conv1(x))  # [B, 64, 160, 160]
#         x = self.leaky_relu(self.conv2(x))  # [B, 128, 40, 40]
#         x = self.leaky_relu(self.conv3(x))  # [B, 256, 10, 10]
#         x = self.conv4(x)                   # [B, 1, 1, 1]
#         return x