import torchvision.transforms as transforms
from model.model_args import args

from torchvision import datasets
import torch

class DataLoaderCreator():
    def __init__(self, img_base_dir, img_blur_dir):
        norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transf = transforms.Compose([
            transforms.Resize((args['img_size'],args['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(*norm,inplace=True),
        ])
        dataset_base = datasets.ImageFolder(root=img_base_dir,transform=transf)
        self.dataloader_base = torch.utils.data.DataLoader(dataset_base, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)
        dataset_blurred = datasets.ImageFolder(root=img_blur_dir,transform=transf)
        self.dataloader_blurred = torch.utils.data.DataLoader(dataset_blurred, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)