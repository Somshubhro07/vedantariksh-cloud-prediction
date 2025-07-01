# scripts/inference.py
import torch
import argparse
import sys
import os
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.logger import Logger
from utils.inference import CloudPredictor
from models.unet_conditional import ConditionalUNet
from models.diffusion_conditional import ConditionalDiffusion

def load_input_images(image_paths, image_size=(256, 256)):
    """Load and preprocess input images"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = transform(img)
        images.append(img)
    
    return torch.stack(images).unsqueeze(0)  # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description='Cloud Prediction Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input_images', nargs='+', required=True, help='Input image paths')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to predict')
    parser.add_argument('--create_gif', action='store_true', help='Create animated GIF')
    
    args = parser.parse_args()
    
    # Initialize config and logger
    config = Config()
    logger = Logger("CloudInference", config.log_dir)
    
    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("Loading models...")
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
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize predictor
    predictor = CloudPredictor(unet, diffusion, device, logger)
    
    # Load input images
    logger.info(f"Loading input images: {args.input_images}")
    input_frames = load_input_images(args.input_images, config.data.image_size)
    input_frames = input_frames.to(device)
    
    # Generate predictions
    logger.info(f"Generating {args.num_frames} future frames...")
    predictions = predictor.predict_sequence(input_frames, args.num_frames)
    
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    predictor.save_predictions(predictions, args.output_dir)
    
    # Create GIF if requested
    if args.create_gif:
        gif_path = os.path.join(args.output_dir, 'prediction_sequence.gif')
        predictor.create_gif(predictions, gif_path)
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()
