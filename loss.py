import torch.nn as nn
import torchvision.models as models
import numpy as np


from torch.autograd import Variable, grad

import torch.nn.functional as F
import torch

from model_args import args

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

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    squeezed_tensor = d_interpolates.view(args['batch_size'], -1)
    # Get gradient w.r.t. interpolates
    gradients = grad(
        outputs=squeezed_tensor,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty