import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import os

def save_imgs(model, imgs_blur, epoch):
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
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/img_pair_epoch_{epoch}.png')