import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CalibrationDataset(Dataset):
    def __init__(self, image_paths, image_width, image_height):
        self.image_paths = image_paths
        self.image_width = image_width
        self.image_height = image_height
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))
        tensor = self.transform(Image.fromarray(image)).to(dtype=torch.float32)

        return tensor