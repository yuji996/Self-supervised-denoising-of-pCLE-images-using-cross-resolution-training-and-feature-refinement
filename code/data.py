import torch
import utils
import numpy as np
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
class ImgTrain(data.Dataset):
    def __init__(self, img_dir, patch_size, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_names[item])
        image = Image.open(img_path).convert('L')
        image = self.augment(image)
        if self.transform:
            image = self.transform(image)
        image = np.array(image) / 255.0
        if image.ndim == 3:
            image = image.squeeze(0)
        H, W = image.shape
        ox = np.random.randint(low=0, high=H - self.patch_size)
        oy = np.random.randint(low=0, high=W - self.patch_size)
        image = image[ox:ox + self.patch_size, oy:oy + self.patch_size]
        image = torch.from_numpy(image).unsqueeze(0).float()
        return image

class ImgVal(data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_names[item])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        image = np.array(image) / 255.0
        if image.ndim == 3:
            image = image.squeeze(0)
        image = torch.from_numpy(image).unsqueeze(0).float()
        return image

def loader_train(img_dir, patch_size, batch_size):
    return data.DataLoader(
        dataset=ImgTrain(img_dir, patch_size),
        batch_size=batch_size,
        shuffle=True
    )

def loader_val(img_dir, batch_size):
    return data.DataLoader(
        dataset=ImgVal(img_dir),
        batch_size=batch_size,
        shuffle=False
    )
