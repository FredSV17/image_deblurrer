import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
import torch


import torch
import torch.nn as nn
import torchvision.models as models


import torch
import torch.nn as nn

    
class LightUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=32):
        super(LightUNetGenerator, self).__init__()

        # Encoder
        self.down1 = self.conv_block(in_channels, features)
        self.down2 = self.conv_block(features, features * 2)

        # Bottleneck
        self.bottleneck = self.conv_block(features * 2, features * 4)  # 160 -> 80

        # Decoder
        self.up1 = self.up_block(features * 4, features * 2)          # 80 -> 160
        self.up2 = self.up_block(features * 2 * 2, features)          # 160 -> 320

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
    


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()

        # DeblurGAN uses PatchGAN-style convolutions (kernel=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1)      # ↓2
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1)     # ↓2
        self.bn2   = nn.BatchNorm2d(features * 2)

        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1) # ↓2
        self.bn3   = nn.BatchNorm2d(features * 4)

        self.conv4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1) # ↓1
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