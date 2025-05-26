import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class SafetyGearDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))
        self.image_names = self.annotations['image_name'].values
        self.labels = self.annotations[['Glasses', 'NoGlasses', 'Helmet', 'NoHelmet', 'Mask', 'NoMask', 'Rompi', 'NoRompi']].values

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label