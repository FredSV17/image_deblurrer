import torch.nn as nn

class BaseBlocks(nn.Module):
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            self.norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            self.norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )

    def conv_transposed_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            self.norm_layer(out_channels),
            nn.ReLU()
        )
        
    def conv_upsample_block(self, in_channels, out_channels):
        # Convolution after upsample
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            self.norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )