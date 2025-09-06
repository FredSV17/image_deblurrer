import os

from torch.autograd import Variable

import torch

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torch.nn import L1Loss


from model.model_args import args
from loss import compute_gradient_penalty, PerceptualLoss
    
from model.gan_framework import GAN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


os.makedirs("results/images", exist_ok=True)




    
def train_wgan(model, dtl, show_results_by_epoch=5, save_model_by_epoch=False):

    # ----------
    #  Training
    # ----------
    percept_loss = PerceptualLoss()
    l1 = L1Loss()
    batches_done = 0
    for epoch in range(args['n_epochs']):
        for i, ((imgs_normal, _), (imgs_blur, _)) in enumerate(zip(dtl.dataloader_base, dtl.dataloader_blurred)):
            imgs_normal = imgs_normal.to(device)
            imgs_blur = imgs_blur.to(device)
            # Configure input
            real_imgs = Variable(imgs_normal.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            
            # Sample noise as generator input
            # Generate a batch of images
            fake_imgs = model.generator(imgs_blur).detach()
            model.optimizer_D.zero_grad()
            # Real images
            real_validity = model.discriminator(real_imgs)
            # Fake images
            fake_validity = model.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(model.discriminator, real_imgs.data, fake_imgs.data)
            
            # Adversarial loss
            loss_D = -(torch.mean(real_validity) - torch.mean(fake_validity)) + gradient_penalty
            
            loss_D.backward()
            model.optimizer_D.step()

            # Train the generator every n_critic iterations
            if i % args['n_critic'] == 0:

                # -----------------
                #  Train Generator
                # -----------------

                model.optimizer_G.zero_grad()
                
                # Generate a batch of images
                gen_imgs = model.generator(imgs_blur)
                # Adversarial loss
                loss_adv = -torch.mean(model.discriminator(gen_imgs))
                
                l1_loss = l1(gen_imgs, imgs_normal)
                content_loss = percept_loss.get_loss(gen_imgs, imgs_normal)
                # Combined loss
                loss_G = l1_loss + 0.1 * loss_adv + 0.1 * content_loss
                loss_G.backward()
                model.optimizer_G.step()
                
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Critic score(Fake): %.4f] [Critic score(True): %.4f] [D loss: %.4f] [G loss: %.4f]" %
                (epoch, args['n_epochs'], batches_done % len(dtl.dataloader_base), len(dtl.dataloader_blurred), 
                 torch.mean(fake_validity).item(), torch.mean(real_validity).item(), loss_D.item(), loss_G.item()),
                end='', flush=True
            )
            
            if batches_done % args['sample_interval'] == 0:
                save_image(gen_imgs.data[:25], "results/images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1
            
            
            # Generate and store a grid of fake images every 5 epochs
        if (epoch + 1) % show_results_by_epoch == 0 and show_results_by_epoch != False:
            with torch.no_grad():
                deblurred_images = model.generator(imgs_blur)
            #img_list.append(vutils.make_grid(denoised_images, padding=2, normalize=True))
            f, ax = plt.subplots(2,1, figsize=(10, 10))
            ax[0].axis('off')
            ax[0].set_title("Generated images")
            
            ax[0].imshow(make_grid(deblurred_images.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            
            ax[1].imshow(make_grid(imgs_blur.detach().cpu(), padding=2, normalize=True).permute(1, 2, 0))
            ax[1].set_title("Blur images")
            
            ax[1].axis('off')
            if not os.path.exists('results/images'):
                os.makedirs('results/images')
            plt.savefig(f'results/images/img_pair_epoch_{epoch}.png')
            
            if save_model_by_epoch:
                print("Saving model...")
                model.save_model(epoch)
        print()