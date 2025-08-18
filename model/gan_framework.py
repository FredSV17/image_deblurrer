from model.blocks import DeepUNetBlock, LightDiscriminator
import torch
from model.model_args import args
import os

class GAN():
    def __init__(self, saved_model_path, device):
        # Initialize generator + discriminator
        self.generator = DeepUNetBlock()
        self.discriminator = LightDiscriminator()
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=args['lr_gen'])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args['lr_dis'])
        
        self.saved_model_path = saved_model_path
        
        if device == 'cuda':
            self.generator.cuda()
            self.discriminator.cuda()
             
    def save_model(self, epoch):
        if not os.path.exists(self.saved_model_path):
            os.makedirs(self.saved_model_path)
        # Saving
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.optimizer_G.state_dict(),
            'd_optimizer_state_dict': self.optimizer_D.state_dict(),
            'epoch': epoch
        }, os.path.join(self.saved_model_path, 'wgan_checkpoint.pth'))