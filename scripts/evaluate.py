# scripts/evaluate.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import sys
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.dataset import CloudSequenceDataset
from utils.logger import Logger
from models.unet_conditional import ConditionalUNet
from models.diffusion_conditional import ConditionalDiffusion

def calculate_metrics(pred, target):
    """Calculate evaluation metrics"""
    # Convert to numpy
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Denormalize
    pred_np = (pred_np + 1) / 2
    target_np = (target_np + 1) / 2
    
    # Clip values
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Calculate metrics
    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))
    
    # SSIM and PSNR
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    
    return {
        'mse': mse,
        'mae': mae,
        'ssim': ssim_val,
        'psnr': psnr_val
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Cloud Prediction Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, default='data/samples', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize config and logger
    config = Config()
    config.data.root_dir = args.data_dir
    logger = Logger("CloudEvaluation", config.log_dir)
    
    # Setup device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = CloudSequenceDataset(
        root_dir=config.data.root_dir,
        input_frames=config.data.input_frames,
        output_frames=config.data.output_frames,
        image_size=config.data.image_size,
        augmentation=False,  # No augmentation for evaluation
        normalize=config.data.normalize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
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
    unet.eval()
    
    # Evaluation
    logger.info("Starting evaluation...")
    all_metrics = []
    
    with torch.no_grad():
        for i, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Prepare conditioning
            cond = input_seq.view(input_seq.shape[0], -1, input_seq.shape[-2], input_seq.shape[-1])
            
            batch_metrics = []
            
            # Evaluate each target frame
            for j in range(target_seq.shape[1]):
                target = target_seq[:, j]
                
                # Generate prediction
                shape = (1, target.shape[-2], target.shape[-1])
                pred = diffusion.sample(cond, shape)
                
                # Calculate metrics
                metrics = calculate_metrics(pred[0], target[0])
                batch_metrics.append(metrics)
            
            # Average metrics for this batch
            avg_metrics = {
                key: np.mean([m[key] for m in batch_metrics])
                for key in batch_metrics[0].keys()
            }
            
            all_metrics.append(avg_metrics)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(dataloader)} batches")
    
    # Calculate final metrics
    final_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"MSE: {final_metrics['mse']:.6f}")
    logger.info(f"MAE: {final_metrics['mae']:.6f}")
    logger.info(f"SSIM: {final_metrics['ssim']:.6f}")
    logger.info(f"PSNR: {final_metrics['psnr']:.2f} dB")
    
    # Save results
    results_path = os.path.join(config.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Cloud Prediction Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Total samples: {len(dataset)}\n\n")
        f.write("Metrics:\n")
        f.write(f"MSE: {final_metrics['mse']:.6f}\n")
        f.write(f"MAE: {final_metrics['mae']:.6f}\n")
        f.write(f"SSIM: {final_metrics['ssim']:.6f}\n")
        f.write(f"PSNR: {final_metrics['psnr']:.2f} dB\n")
    
    logger.info(f"Results saved to: {results_path}")
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()