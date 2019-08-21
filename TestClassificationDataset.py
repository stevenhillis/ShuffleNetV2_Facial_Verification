import os

import torch
from PIL import Image
import torchvision
from torch.utils import data

class TestClassificationDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.labels = []
        images = os.listdir(root)
        for image in images:
            self.labels.append(int(image.split(".")[0]))
        self.labels.sort()
        print(*self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file = self.root + repr(self.labels[index]) + ".jpg"
        image = Image.open(file)
        image = torchvision.transforms.ToTensor()(image)
        return image
