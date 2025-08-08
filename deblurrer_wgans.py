import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.autograd as autograd

from generator_discriminator_arch import LightUNetGenerator, Discriminator
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


os.makedirs("images", exist_ok=True)

args = {
    'n_epochs': 50,
    'batch_size': 32,
    'lr_gen': 1e-4,
    'lr_dis': 1e-4,
    'n_cpu': 8,
    'latent_dim': 64,
    'img_size': 640,
    'channels': 3,
    'n_critic': 3,
    'clip_value': 0.01,
    'sample_interval': 400,
}

img_shape = (args['channels'], args['img_size'], args['img_size'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize generator and discriminator
generator = LightUNetGenerator()
discriminator = Discriminator()

if device == 'cuda':
    generator.cuda()
    discriminator.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
#     ),
#     batch_size=args['batch_size'],
#     shuffle=True,
# )


# Loss weight for gradient penalty
lambda_gp = 10

norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transf = transforms.Compose([
    transforms.Resize((args['img_size'],args['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(*norm,inplace=True),
])




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
    gradients = autograd.grad(
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

def scale_image(image):
    """
    Scales an image with values in the range [0, 50] to the range [-1, 1].
    """
    # Ensure the input is a numpy array
    image = np.array(image, dtype=np.float32)
    
    # Apply the linear scaling transformation
    scaled_image = (image * 2 / 50) - 1
    
    return Tensor(scaled_image)

def channel_consistency_loss(img):
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return torch.mean(torch.abs(r - g)) + torch.mean(torch.abs(r - b)) + torch.mean(torch.abs(g - b))

def train_wgan(img_nrml_dir, img_blur_dir, show_results_by_epoch=5):
    dataset_normal = datasets.ImageFolder(root=img_nrml_dir,transform=transf)
    dataloader = torch.utils.data.DataLoader(dataset_normal, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    dataset_noisy = datasets.ImageFolder(root=img_blur_dir,transform=transf)
    dataloader_noisy = torch.utils.data.DataLoader(dataset_noisy, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args['lr_gen'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args['lr_dis'])
    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(args['n_epochs']):
        for i, ((imgs_normal, _), (imgs_noisy, _)) in enumerate(zip(dataloader, dataloader_noisy)):
            imgs_normal = imgs_normal
            imgs_noisy = imgs_noisy
            imgs_normal = imgs_normal.to(device)
            imgs_noisy = imgs_noisy.to(device)
            # Configure input
            real_imgs = Variable(imgs_normal.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            # Generate a batch of images
            fake_imgs = generator(imgs_noisy).detach()
            
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            
            # Adversarial loss
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp # * gradient_penalty
            
            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args['clip_value'], args['clip_value'])

            # Train the generator every n_critic iterations
            if i % args['n_critic'] == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(imgs_noisy)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs)) 
                l1_loss = F.l1_loss(gen_imgs, imgs_normal)
                #channel_loss = channel_consistency_loss(gen_imgs)
                # Combined loss
                loss_G = loss_G + 0.1 * l1_loss
                loss_G.backward()
                optimizer_G.step()
                
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" %
                (epoch, args['n_epochs'], batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item()),
                end='', flush=True
            )
            if batches_done % args['sample_interval'] == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1
            # Generate and store a grid of fake images every 5 epochs
        if (epoch + 1) % show_results_by_epoch == 0 and show_results_by_epoch != False:
            with torch.no_grad():
                denoised_images = generator(imgs_noisy)
            #img_list.append(vutils.make_grid(denoised_images, padding=2, normalize=True))
            print('Generated images')
            plt.figure(figsize=(10, 10))
            #plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            # Make grid from denoised images (B, C, H, W)
            plt.imshow(vutils.make_grid(denoised_images.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            plt.axis('off')
            plt.title("Generated images")
            plt.show()
            
            plt.figure(figsize=(10, 10))
            plt.title("Noise images")
            plt.imshow(vutils.make_grid(imgs_noisy.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            plt.axis('off')
            plt.show()
        print()