import os

from torch.autograd import Variable

import torch

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt



from GAN_model.model_args import args
from GAN_model.loss import DiscriminatorLoss, GeneratorLoss
    
from GAN_model.gan_framework import GAN
from results.show_results import save_imgs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


os.makedirs("results/images", exist_ok=True)




    
def train_wgan(model, dtl, show_results_by_epoch=5, save_model_by_epoch=False):

    # ----------
    #  Training
    # ----------

    disc_loss_obj = DiscriminatorLoss(True)
    gen_loss_obj = GeneratorLoss()
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

            d_loss = disc_loss_obj.get_loss(model, real_imgs, fake_imgs)
            d_loss.backward()
            model.optimizer_D.step()

            # Train the generator every n_critic iterations
            if i % args['n_critic'] == 0:

                # -----------------
                #  Train Generator
                # -----------------

                model.optimizer_G.zero_grad()
                
                # Generate a batch of images
                gen_imgs = model.generator(imgs_blur)
                
                g_loss = gen_loss_obj.get_loss(model, imgs_normal, gen_imgs)
                g_loss.backward()
                model.optimizer_G.step()
                
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Critic score(Fake): %.4f] [Critic score(True): %.4f] [D loss: %.4f] [G loss: %.4f]" %
                (epoch, args['n_epochs'], batches_done % len(dtl.dataloader_base), len(dtl.dataloader_blurred), 
                 torch.mean(disc_loss_obj.fake_validity).item(), torch.mean(disc_loss_obj.real_validity).item(), d_loss.item(), g_loss.item()),
                end='', flush=True
            )
            
            batches_done += 1
            
            
        if (epoch + 1) % show_results_by_epoch == 0 and show_results_by_epoch != False:
            print("Saving images...")
            save_imgs(model, imgs_blur, epoch)
            
            if save_model_by_epoch:
                print("Saving model...")
                model.save_model(epoch)
        print()