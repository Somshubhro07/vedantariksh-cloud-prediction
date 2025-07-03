import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
import glob
import rasterio

class CloudSequenceDataset(Dataset):
    """Dataset for cloud sequence prediction with multi-channel INSAT data"""
    
    def __init__(
        self,
        root_dir: str,
        input_frames: int = 4,
        output_frames: int = 2,
        image_size: Tuple[int, int] = (128, 128),
        augmentation: bool = True,
        normalize: bool = True,
        channels: int = 6
    ):
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.total_frames = input_frames + output_frames
        self.image_size = image_size
        self.channels = channels
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]) if augmentation else None
        
        self.normalize = normalize
        self.mean = torch.tensor([0.5] * channels)
        self.std = torch.tensor([0.5] * channels)
        
        self.sequences = self._find_sequences()
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences found in {root_dir}")
    
    def _find_sequences(self) -> List[List[str]]:
        sequences = []
        all_files = glob.glob(os.path.join(self.root_dir, "*.tif"))
        all_files.sort()
        
        timestamp_groups = {}
        for file in all_files:
            timestamp = os.path.basename(file).split('_')[1] + '_' + os.path.basename(file).split('_')[2]
            if timestamp not in timestamp_groups:
                timestamp_groups[timestamp] = []
            timestamp_groups[timestamp].append(file)
        
        for timestamp, files in timestamp_groups.items():
            if len(files) == self.channels:
                sequences.append(files)
        
        valid_sequences = []
        for i in range(len(sequences) - self.total_frames + 1):
            valid_sequences.append(sequences[i:i + self.total_frames])
        return valid_sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_files = self.sequences[idx]
        images = []
        
        for frame_group in sequence_files:
            frame_channels = []
            for file_path in frame_group:
                with rasterio.open(file_path) as src:
                    band = src.read(1)
                    band = np.transpose(band, (1, 0))
                    band = band.astype(np.float32) / 65535.0
                    frame_channels.append(torch.from_numpy(band))
            frame = torch.stack(frame_channels)
            
            frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
            
            if self.augment_transform:
                frame = self.augment_transform(frame.unsqueeze(0)).squeeze(0)
            
            if self.normalize:
                frame = (frame - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
            
            images.append(frame)
        
        input_seq = torch.stack(images[:self.input_frames])
        target_seq = torch.stack(images[self.input_frames:])
        
        return input_seq, target_seq