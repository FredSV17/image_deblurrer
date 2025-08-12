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
        self.down1 = self.conv_block(in_channels, features)          # 640 -> 320
        self.down2 = self.conv_block(features, features * 2)         # 320 -> 160

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

        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=4, stride=4, padding=0)   # 640 → 160
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=4, padding=0)   # 160 → 40
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=4, padding=0)  # 40 → 10
        self.conv4 = nn.Conv2d(features * 4, 1, kernel_size=10, stride=1, padding=0)         # 10 → 1

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  # [B, 64, 160, 160]
        x = self.leaky_relu(self.conv2(x))  # [B, 128, 40, 40]
        x = self.leaky_relu(self.conv3(x))  # [B, 256, 10, 10]
        x = self.conv4(x)                   # [B, 1, 1, 1]
        return x
    
    
# class UNetGenerator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, features=16):
#         super(UNetGenerator, self).__init__()

#         # Encoder
#         self.down1 = self.conv_block(in_channels, features)
#         self.down2 = self.conv_block(features, features * 2)
#         self.down3 = self.conv_block(features * 2, features * 4)
#         self.down4 = self.conv_block(features * 4, features * 8)

#         # Bottleneck
#         self.bottleneck = self.conv_block(features * 8, features * 16)
        
#         # Decoder
#         self.up4 = self.up_block(features * 16, features * 8)
#         self.up3 = self.up_block(features * 8 * 2, features * 4)  # x2 for skip conn
#         self.up2 = self.up_block(features * 4 * 2, features * 2)
#         self.up1 = self.up_block(features * 2 * 2, features)
#         self.up0 = self.up_block(features * 2, features * 2)
        
#         # Final
#         self.final = nn.Conv2d(features * 2, out_channels, kernel_size=3, padding=1)
#         self.tanh = nn.Tanh()

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm2d(out_channels),
#             nn.LeakyReLU(0.2)
#         )
        
        
#     def up_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm2d(out_channels),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         # Encoder
#         d1 = self.down1(x)  # 64 -> 32
#         d2 = self.down2(d1) # 32 -> 16
#         d3 = self.down3(d2) # 16 -> 8
#         d4 = self.down4(d3) # 8 -> 4

#         # Bottleneck
#         bottleneck = self.bottleneck(d4)  # 4 -> 2

#         # Decoder with skip connections
#         up4 = self.up4(bottleneck)        # 2 -> 4
#         # # Test: add noise to the decoder
#         noise = torch.randn_like(up4)
#         up4 = up4 + noise * 0.1  # scale factor to control randomness
        
#         up4 = torch.cat([up4, d4], dim=1)

#         up3 = self.up3(up4)               # 4 -> 8
#         up3 = torch.cat([up3, d3], dim=1)

#         up2 = self.up2(up3)               # 8 -> 16
#         up2 = torch.cat([up2, d2], dim=1)

#         up1 = self.up1(up2)               # 16 -> 32
#         up1 = torch.cat([up1, d1], dim=1)
        
#         up0 = self.up0(up1)               # 32 -> 64
#         out = self.final(up0)
#         out = self.tanh(out)
#         # Clamping so the results will not be oversaturated
#         return torch.clamp(out, -0.98, 0.98)
    
# class Discriminator(nn.Module):
#     def __init__(self, features=16):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, features, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv5 = nn.Conv2d(features * 8, features * 16, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv6 = nn.Conv2d(features * 16, features * 8, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv7 = nn.Conv2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv8 = nn.Conv2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, input):
#         x = self.leaky_relu(self.conv1(input))
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))
#         x = self.leaky_relu(self.conv4(x))
#         x = self.leaky_relu(self.conv5(x))
#         x = self.leaky_relu(self.conv6(x))
#         x = self.leaky_relu(self.conv7(x))
#         x = self.conv8(x)
#         return x