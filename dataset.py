from glob import glob
from PIL import Image

import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

EXTENSION = ['png', 'jpg', 'JPG', 'PNG']
IMAGENET_STAT = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, image_size=224, mean=None, std=None) -> None:
        self.image_path = list()
        for ext in EXTENSION:
            self.image_path += glob(os.path.join(data_dir, f'*.{ext}'))
        self.image_size = image_size
        if mean is None or std is None:
            print('Use ImageNet stats')
            self.m = torch.tensor(IMAGENET_STAT[0])
            self.std = torch.tensor(IMAGENET_STAT[1])
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).resize((self.image_size, self.image_size))
        image = torch.tensor(np.array(image)) / 255.
        image = (image-self.m) / self.std
        return image
    
def get_loader(dataset_dir, image_size, batch_size=50, num_workers=8, mean=None, std=None):
    dataset = ImageFolderDataset(dataset_dir, image_size=image_size, mean=mean, std=std)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        drop_last=False)
    return loader