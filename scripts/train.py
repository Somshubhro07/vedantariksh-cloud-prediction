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
    parser.add_argument('--data_dir', type=str, default='data/samples', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.data.root_dir = args.data_dir
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    
    # Initialize logger
    logger = Logger("CloudTraining", config.log_dir)
    logger.info("Starting cloud prediction training")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = CloudSequenceDataset(
        root_dir=config.data.root_dir,
        input_frames=config.data.input_frames,
        output_frames=config.data.output_frames,
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
        normalize=config.data.normalize
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Dataset size - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    logger.info("Initializing models...")
    unet = ConditionalUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_groups=config.model.num_groups,
        attention_resolutions=config.model.attention_resolutions,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention
    ).to(device)
    
    diffusion = ConditionalDiffusion(
        model=unet,
        timesteps=config.diffusion.timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule_type=config.diffusion.schedule_type
    ).to(device)
    
    # Initialize trainer
    trainer = Trainer(unet, diffusion, train_loader, val_loader, config, logger)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
