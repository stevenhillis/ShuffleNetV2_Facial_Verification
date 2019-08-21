import os

from torch.utils import data
from PIL import Image
import torchvision

class TestVerificationDataset(data.Dataset):
    def __init__(self, root, trials_filename):
        self.root = root
        self.trials = []
        self.trials_filename = trials_filename
        with open(self.trials_filename, "r") as trials_file:
            for line in trials_file:
                file_a, file_b = line.split()
                self.trials.append([int(file_a.split(".")[0]), int(file_b.split(".")[0])])

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        file_a, file_b = self.trials[index]
        file_a = self.root + repr(file_a) + ".jpg"
        file_b = self.root + repr(file_b) + ".jpg"
        image_a = Image.open(file_a)
        image_b = Image.open(file_b)
        image_a = torchvision.transforms.ToTensor()(image_a)
        image_b = torchvision.transforms.ToTensor()(image_b)
        return [image_a, image_b]

    def get_trial(self, index):
        return self.trials[index]
