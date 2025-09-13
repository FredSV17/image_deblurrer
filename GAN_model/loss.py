import torch.nn as nn
import torchvision.models as models
import numpy as np


from torch.autograd import Variable, grad
from torch.nn import L1Loss

import torch.nn.functional as F
import torch

from GAN_model.model_args import args

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
            
    def get_loss(self, fake_samples, real_samples):
        f_fake = self.contentFunc.forward(fake_samples)
        f_real = self.contentFunc.forward(real_samples)
        f_real_no_grad = f_real.detach()
        loss_content = ((f_fake - f_real_no_grad)**2).mean()
        return loss_content

class GeneratorLoss():
    def __init__(self):    
        self.percept_loss = PerceptualLoss()
        self.l1 = L1Loss()
        self.l1_weight = 0.1
        self.adv_weight = 1
        self.content_weight = 0.1
        
    def get_loss(self, model, real_imgs, gen_imgs):
        # Adversarial loss
        loss_adv = -torch.mean(model.discriminator(gen_imgs))
        # l1 loss
        l1_loss = self.l1(gen_imgs, real_imgs)
        # Perceptual loss
        content_loss = self.percept_loss.get_loss(gen_imgs, real_imgs)
        # Combined loss
        return self.l1_weight + l1_loss + self.adv_weight + loss_adv + self.content_weight * content_loss
        
class DiscriminatorLoss():
    
    def __init__(self, has_gp=False):
        self.has_gp = has_gp
        # Loss weight for gradient penalty
        self.lambda_gp = 10
        
    def get_loss(self, model, real_imgs, fake_imgs):
        # Real images
        self.real_validity = model.discriminator(real_imgs)
        # Fake images
        self.fake_validity = model.discriminator(fake_imgs)
        
        adv_loss = -(torch.mean(self.real_validity) - torch.mean(self.fake_validity))
        
        if self.has_gp:
            gradient_penalty = self.compute_gradient_penalty(model.discriminator, real_imgs.data, fake_imgs.data)
            return adv_loss + self.lambda_gp * gradient_penalty
        else:
            return adv_loss
        
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
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
        return ((gradients_norm - 1) ** 2).mean()