import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt

from generator_discriminator_arch import LightUNetGenerator, Discriminator
from model_args import args
from loss import compute_gradient_penalty
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


os.makedirs("images", exist_ok=True)


img_shape = (args['channels'], args['img_size'], args['img_size'])

# Initialize generator and discriminator
generator = LightUNetGenerator()
discriminator = Discriminator()

if device == 'cuda':
    generator.cuda()
    discriminator.cuda()

# Loss weight for gradient penalty
lambda_gp = 10

norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transf = transforms.Compose([
    transforms.Resize((args['img_size'],args['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(*norm,inplace=True),
])

def save_model(optimizer_G, optimizer_D, epoch):
    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Saving
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': optimizer_G.state_dict(),
        'd_optimizer_state_dict': optimizer_D.state_dict(),
        'epoch': epoch
    }, os.path.join(model_path, 'wgan_checkpoint.pth'))
    
def train_wgan(img_nrml_dir, img_blur_dir, show_results_by_epoch=5, save_model_by_epoch=False):
    dataset_normal = datasets.ImageFolder(root=img_nrml_dir,transform=transf)
    dataloader = torch.utils.data.DataLoader(dataset_normal, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    dataset_blurred = datasets.ImageFolder(root=img_blur_dir,transform=transf)
    dataloader_blurred = torch.utils.data.DataLoader(dataset_blurred, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args['lr_gen'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args['lr_dis'])
    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(args['n_epochs']):
        for i, ((imgs_normal, _), (imgs_noisy, _)) in enumerate(zip(dataloader, dataloader_blurred)):
            imgs_normal = imgs_normal
            imgs_noisy = imgs_noisy
            imgs_normal = imgs_normal.to(device)
            imgs_noisy = imgs_noisy.to(device)
            # Configure input
            real_imgs = Variable(imgs_normal.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            
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
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
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
            plt.savefig(f'images/Results/Gen_imgs_epoch_{epoch}.png')
            
            plt.figure(figsize=(10, 10))
            plt.title("Blur images")
            plt.imshow(vutils.make_grid(imgs_noisy.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(f'images/Results/Blur_imgs_epoch_{epoch}.png')
            if save_model_by_epoch:
                print("Saving model...")
                save_model(optimizer_G, optimizer_D, epoch)
