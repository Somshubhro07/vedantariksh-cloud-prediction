# scripts/train.py
import torch
from torch.utils.data import DataLoader, random_split
import argparse
import sys
import os

# Add project root to path
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
    
    # Create a small sample dataset first to understand the tensor shapes
    sample_dataset = CloudSequenceDataset(
        root_dir=config.data.root_dir,
        input_frames=config.data.input_frames,
        output_frames=config.data.output_frames,
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    # Get a sample to understand the actual tensor shapes
    sample_input, sample_target = sample_dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample target shape: {sample_target.shape}")
    
    # Calculate input channels based on actual tensor shapes
    if len(sample_input.shape) == 5:
        # Format: [input_frames, batch_or_sequence, channels, H, W]
        input_frames, batch_dim, channels, height, width = sample_input.shape
        conditioning_channels = input_frames * channels  # This will be 4 * 6 = 24
    elif len(sample_input.shape) == 4:
        # Format: [input_frames, channels, H, W]
        input_frames, channels, height, width = sample_input.shape
        conditioning_channels = input_frames * channels
    else:
        raise ValueError(f"Unexpected sample input shape: {sample_input.shape}")
    
    # Calculate target channels
    if len(sample_target.shape) == 5:
        # Format: [output_frames, batch_or_sequence, channels, H, W]
        output_frames, batch_dim, target_channels, height, width = sample_target.shape
    elif len(sample_target.shape) == 4:
        # Format: [output_frames, channels, H, W]
        output_frames, target_channels, height, width = sample_target.shape
    else:
        raise ValueError(f"Unexpected sample target shape: {sample_target.shape}")
    
    # UNet input channels = conditioning_channels + target_channels (for concatenation)
    # During training: conditioning (24) + noisy_target (6) = 30 channels
    unet_input_channels = conditioning_channels + target_channels
    unet_output_channels = target_channels  # UNet predicts noise, same shape as target
    
    config.model.in_channels = unet_input_channels
    config.model.out_channels = unet_output_channels
    config.training.epochs = 500  # Reduced for initial run
    
    logger = Logger("CloudTraining", config.log_dir)
    logger.info(f"Sample input shape: {sample_input.shape}")
    logger.info(f"Sample target shape: {sample_target.shape}")
    logger.info(f"Conditioning channels: {conditioning_channels}")
    logger.info(f"Target channels: {target_channels}")
    logger.info(f"UNet input channels: {config.model.in_channels}")
    logger.info(f"UNet output channels: {config.model.out_channels}")
    
    device = torch.device(config.device)
    
    # Create full dataset
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
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
    
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