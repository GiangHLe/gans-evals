from glob import glob
from PIL import Image

import numpy as np
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

EXTENSION = ['png', 'jpg', 'JPG', 'PNG']
IMAGENET_STAT = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
STATISTIC_STAT = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, image_size=224, mean_cov_imagenet=False, transform=False) -> None:
        self.image_path = list()
        for ext in EXTENSION:
            self.image_path += glob(os.path.join(data_dir, f'*.{ext}'))
        self.image_size = image_size
        if mean_cov_imagenet:
            stat = IMAGENET_STAT
        else:
            stat = STATISTIC_STAT
        self.m = np.array(stat[0])
        self.std = np.array(stat[1])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = np.array(image) / 255.
        if self.transform:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            image = (image-self.m[None,None,:]) / self.std[None,None,:]
        return torch.tensor(image.astype(np.float32)).permute(2, 0, 1)
    
def get_loader(dataset_dir, image_size, batch_size=50, num_workers=8, imagenet_stat=False, transform=False):
    dataset = ImageFolderDataset(dataset_dir, image_size=image_size, mean_cov_imagenet=imagenet_stat, transform=transform)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        drop_last=False)
    return loader