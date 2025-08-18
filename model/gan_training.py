import os

from torch.autograd import Variable

import torch

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


from model.model_args import args
from loss import compute_gradient_penalty, PerceptualLoss
    
from model.gan_framework import GAN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


os.makedirs("results/images", exist_ok=True)

# Loss weight for gradient penalty
lambda_gp = 10


    
def train_wgan(model, dtl, show_results_by_epoch=5, save_model_by_epoch=False):

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(args['n_epochs']):
        for i, ((imgs_normal, _), (imgs_noisy, _)) in enumerate(zip(dtl.dataloader_base, dtl.dataloader_blurred)):
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
            fake_imgs = model.generator(imgs_noisy).detach()
            
            # Real images
            real_validity = model.discriminator(real_imgs)
            # Fake images
            fake_validity = model.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(model.discriminator, real_imgs.data, fake_imgs.data)
            
            # Adversarial loss
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            loss_D.backward()
            model.optimizer_D.step()

            # Clip weights of discriminator
            for p in model.discriminator.parameters():
                p.data.clamp_(-args['clip_value'], args['clip_value'])

            # Train the generator every n_critic iterations
            if i % args['n_critic'] == 0:

                # -----------------
                #  Train Generator
                # -----------------

                model.optimizer_G.zero_grad()
                percept_loss = PerceptualLoss()
                # Generate a batch of images
                gen_imgs = model.generator(imgs_noisy)
                # Adversarial loss
                loss_G = -torch.mean(model.discriminator(gen_imgs))
                # l1_loss = F.l1_loss(gen_imgs, imgs_normal)
                content_loss = percept_loss.get_loss(gen_imgs, imgs_normal)
                # Combined loss
                loss_G = loss_G + 0.1 * content_loss
                loss_G.backward()
                model.optimizer_G.step()
                
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" %
                (epoch, args['n_epochs'], batches_done % len(dtl.dataloader_base), len(dtl.dataloader_blurred), loss_D.item(), loss_G.item()),
                end='', flush=True
            )
            
            if batches_done % args['sample_interval'] == 0:
                save_image(gen_imgs.data[:25], "results/images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1
            
            
            # Generate and store a grid of fake images every 5 epochs
        if (epoch + 1) % show_results_by_epoch == 0 and show_results_by_epoch != False:
            with torch.no_grad():
                denoised_images = model.generator(imgs_noisy)
            #img_list.append(vutils.make_grid(denoised_images, padding=2, normalize=True))
            print('Generated images')
            plt.figure(figsize=(10, 10))

            plt.imshow(make_grid(denoised_images.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            plt.axis('off')
            plt.title("Generated images")
            if not os.path.exists('results/images/Gen_imgs'):
                os.makedirs('results/images/Gen_imgs')
            plt.savefig(f'results/images/Gen_imgs/Gen_imgs_epoch_{epoch}.png')
            plt.close()
            
            plt.figure(figsize=(10, 10))
            plt.imshow(make_grid(imgs_noisy.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            plt.title("Blur images")
            
            plt.axis('off')
            if not os.path.exists('results/images/Blr_imgs'):
                os.makedirs('results/images/Blr_imgs')
            plt.savefig(f'results/images/Blr_imgs/Blr_imgs_epoch_{epoch}.png')
            plt.close()
            
            if save_model_by_epoch:
                print("Saving model...")
                model.save_model(epoch)
