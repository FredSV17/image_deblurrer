import torch.nn as nn
import torchvision.models as models
import numpy as np


from torch.autograd import Variable, grad

import torch.nn.functional as F
import torch

from model.model_args import args

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

class PerceptualLoss():
    
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model
		
    def __init__(self):
        self.contentFunc = self.contentFunc()
            
    def get_loss(self, fakeIm, realIm):
        fakeIm = F.interpolate(fakeIm, size=(224,224), mode='bilinear', align_corners=False)
        realIm = F.interpolate(realIm, size=(224,224), mode='bilinear', align_corners=False)
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss_content = ((f_fake - f_real_no_grad)**2).mean()
        return loss_content


def compute_gradient_penalty(D, real_samples, fake_samples, device='cuda'):
    """Calculates the gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)

    # Calculate gradients of D(interpolates) w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty