import torch
from torch.utils.data import DataLoader, random_split
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.dataset import CloudSequenceDataset
from utils.logger import Logger
from utils.training import Trainer
from models.unet_conditional import ConditionalUNet
from models.diffusion_conditional import ConditionalDiffusion

def main():
    parser = argparse.ArgumentParser(description='Train Cloud Prediction Model')
    parser.add_argument('--data_dir', type=str, default='data/insat_data', help='Data directory')
    args = parser.parse_args()
    
    config = Config()
    config.data.root_dir = args.data_dir
    
    sample_dataset = CloudSequenceDataset(
        root_dir=config.data.root_dir,
        input_frames=config.data.input_frames,
        output_frames=config.data.output_frames,
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    sample_input, sample_target = sample_dataset[0]
    
    # Calculate channels based on actual tensor processing in training loop
    # Input: [input_frames, 1, channels, H, W] -> [input_frames * channels, H, W]
    conditioning_channels = sample_input.shape[0] * sample_input.shape[2]  # 4 * 6 = 24
    # Target: [output_frames, 1, channels, H, W] -> [channels, H, W] (per frame)
    target_channels = sample_target.shape[2]  # 6
    
    # UNet input will be conditioning + target (concatenated)
    config.model.in_channels = conditioning_channels + target_channels  # 24 + 6 = 30
    config.model.out_channels = target_channels  # 6 (predicting noise for single frame)
    
    logger = Logger("CloudTraining", config.log_dir)
    logger.info(f"Sample input shape: {sample_input.shape}")
    logger.info(f"Sample target shape: {sample_target.shape}")
    logger.info(f"Conditioning channels: {conditioning_channels}")
    logger.info(f"Target channels: {target_channels}")
    logger.info(f"UNet input channels: {config.model.in_channels}")
    logger.info(f"UNet output channels: {config.model.out_channels}")
    
    device = torch.device(config.device)
    
    full_dataset = CloudSequenceDataset(
        root_dir=config.data.root_dir,
        input_frames=config.data.input_frames,
        output_frames=config.data.output_frames,
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    unet = ConditionalUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels
    ).to(device)
    
    diffusion = ConditionalDiffusion(model=unet).to(device)
    trainer = Trainer(unet, diffusion, train_loader, val_loader, config, logger)
    trainer.train()

if __name__ == "__main__":
    main()