# utils/dataset.py

import os
import torch
import torchvision.transforms as T
from PIL import Image

class CloudSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, input_frames=2, output_frames=4, image_size=(256, 256)):
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Scale to [-1, 1]
        ])
        self.file_list = sorted(os.listdir(root_dir))

    def __len__(self):
        return 1  # one sequence for now

    def __getitem__(self, idx):
        input_imgs = []
        target_imgs = []

        for i in range(self.input_frames):
            path = os.path.join(self.root_dir, self.file_list[i])
            img = self.transform(Image.open(path).convert("L"))
            input_imgs.append(img)

        for i in range(self.input_frames, self.input_frames + self.output_frames):
            path = os.path.join(self.root_dir, self.file_list[i])
            img = self.transform(Image.open(path).convert("L"))
            target_imgs.append(img)

        input_tensor = torch.stack(input_imgs)     # Shape: [T, 1, H, W]
        target_tensor = torch.stack(target_imgs)   # Shape: [T, 1, H, W]

        return input_tensor, target_tensor
