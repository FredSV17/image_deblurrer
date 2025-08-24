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

# Loss weight for gradient penalty
lambda_gp = 10

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Gradient penalty for WGAN-GP with PatchGAN discriminator."""
    device = real_samples.device
    batch_size = real_samples.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates)

    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    #gradient_norm = gradients.norm(2, dim=1).mean().data[0]
    
    ########### Logging #############
    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())
    
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()