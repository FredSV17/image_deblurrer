import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
import torch.nn as nn
import torchvision.models as models


import torch
import torch.nn as nn

from model.base_architecture import BaseArchitecture


        
class DeepUNetBlock(BaseArchitecture):
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
        self.up1 = self.up_block(features * 4, features * 4)          # 80 -> 160
        self.up2 = self.up_block(features * 8, features * 4)          # 160 -> 320
        self.up3 = self.up_block(features * 8, features * 2)
        self.up4 = self.up_block(features * 4, features)
        
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






class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()

        # DeblurGAN uses PatchGAN-style convolutions (kernel=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)      # ↓2
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)     # ↓2
        self.bn2   = nn.BatchNorm2d(features * 2)

        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1) # ↓2
        self.bn3   = nn.BatchNorm2d(features * 4)

        self.conv4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=4, padding=1) # ↓1
        self.bn4   = nn.BatchNorm2d(features * 8)

        # Output a 1-channel patch map → PatchGAN
        self.conv5 = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))      # [B, 64,   H/2, W/2]
        x = self.leaky_relu(self.bn2(self.conv2(x)))  # [B, 128,  H/4, W/4]
        x = self.leaky_relu(self.bn3(self.conv3(x)))  # [B, 256,  H/8, W/8]
        x = self.leaky_relu(self.bn4(self.conv4(x)))  # [B, 512,  H/8, W/8]
        x = self.conv5(x)                             # [B, 1,    H/8, W/8]
        return x
    
class LightDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(LightDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)   # 128× 128× 64
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)   # 64× 64× 128
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)  # 32× 32× 256
        self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=4, stride=2, padding=1) # 16× 16x 1    

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  
        x = self.leaky_relu(self.conv2(x))  
        x = self.leaky_relu(self.conv3(x))  
        x = self.conv4(x)                   
        return x