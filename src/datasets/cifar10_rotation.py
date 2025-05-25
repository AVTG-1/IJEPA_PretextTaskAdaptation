import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import random

class CIFAR10Rotation(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=False)
        self.transform = transform

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, _ = self.cifar10[idx]  # original label is ignored
        rotation = random.choice([0, 1, 2, 3])
        rotated_img = self.rotate(img, rotation)
        if self.transform:
            rotated_img = self.transform(rotated_img)
        return rotated_img, rotation

    def rotate(self, img, rotation):
        return img.rotate(90 * rotation)

