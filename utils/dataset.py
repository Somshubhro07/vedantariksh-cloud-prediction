# utils/dataset.py
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
import glob

class CloudSequenceDataset(Dataset):
    """Enhanced dataset for cloud sequence prediction"""
    
    def __init__(
        self,
        root_dir: str,
        input_frames: int = 2,
        output_frames: int = 4,
        image_size: Tuple[int, int] = (256, 256),
        augmentation: bool = True,
        normalize: bool = True
    ):
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.total_frames = input_frames + output_frames
        self.image_size = image_size
        
        # Setup transforms
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        if normalize:
            self.base_transform = transforms.Compose([
                self.base_transform,
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
        
        self.augment_transform = None
        if augmentation:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
        # Find all sequences
        self.sequences = self._find_sequences()
        
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences found in {root_dir}")
    
    def _find_sequences(self) -> List[List[str]]:
        """Find all valid sequences in the dataset"""
        sequences = []
        
        # Look for image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        all_files = []
        
        for ext in image_extensions:
            all_files.extend(glob.glob(os.path.join(self.root_dir, ext)))
            all_files.extend(glob.glob(os.path.join(self.root_dir, "**", ext), recursive=True))
        
        all_files.sort()
        
        # Group files into sequences
        if len(all_files) >= self.total_frames:
            for i in range(len(all_files) - self.total_frames + 1):
                sequence = all_files[i:i + self.total_frames]
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_files = self.sequences[idx]
        
        # Load images
        images = []
        for file_path in sequence_files:
            try:
                image = Image.open(file_path).convert('L')  # Convert to grayscale
                
                # Apply augmentation before base transform
                if self.augment_transform is not None:
                    image = self.augment_transform(image)
                
                image = self.base_transform(image)
                images.append(image)
                
            except Exception as e:
                # If image loading fails, create a random tensor
                images.append(torch.randn(1, *self.image_size))
        
        # Split into input and target sequences
        input_seq = torch.stack(images[:self.input_frames])    # [input_frames, 1, H, W]
        target_seq = torch.stack(images[self.input_frames:])   # [output_frames, 1, H, W]
        
        return input_seq, target_seq