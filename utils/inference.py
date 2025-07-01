# utils/inference.py
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image
import os
from typing import List, Tuple
import numpy as np

class CloudPredictor:
    """Enhanced inference class for cloud prediction"""
    
    def __init__(self, model, diffusion, device, logger):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.logger = logger
        self.to_pil = ToPILImage()
        
        self.model.eval()
    
    @torch.no_grad()
    def predict_sequence(
        self,
        input_frames: torch.Tensor,
        num_frames: int = 4,
        guidance_scale: float = 1.0
    ) -> List[torch.Tensor]:
        """Predict future cloud frames"""
        batch_size = input_frames.shape[0]
        
        # Prepare conditioning
        cond = input_frames.view(batch_size, -1, input_frames.shape[-2], input_frames.shape[-1])
        
        predictions = []
        
        for i in range(num_frames):
            self.logger.info(f"Generating frame {i+1}/{num_frames}")
            
            # Generate frame using diffusion sampling
            shape = (1, cond.shape[-2], cond.shape[-1])
            pred_frame = self.diffusion.sample(cond, shape)
            
            # Clamp and store
            pred_frame = torch.clamp(pred_frame, -1, 1)
            predictions.append(pred_frame)
            
            # Update conditioning for next frame (use last 2 frames)
            if len(predictions) >= 2:
                new_cond = torch.stack([predictions[-2], predictions[-1]], dim=1)
                cond = new_cond.view(batch_size, -1, cond.shape[-2], cond.shape[-1])
        
        return predictions
    
    def save_predictions(
        self,
        predictions: List[torch.Tensor],
        output_dir: str,
        prefix: str = "predicted"
    ):
        """Save predicted frames as images"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            # Convert to PIL
            pred_np = pred.squeeze(0).squeeze(0).cpu()
            pred_np = (pred_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            pred_np = torch.clamp(pred_np, 0, 1)
            
            img = self.to_pil(pred_np)
            
            # Save
            filename = f"{prefix}_frame_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            
            self.logger.info(f"Saved: {filepath}")
    
    def create_gif(
        self,
        predictions: List[torch.Tensor],
        output_path: str,
        duration: int = 500
    ):
        """Create animated GIF from predictions"""
        images = []
        
        for pred in predictions:
            pred_np = pred.squeeze(0).squeeze(0).cpu()
            pred_np = (pred_np + 1) / 2
            pred_np = torch.clamp(pred_np, 0, 1)
            
            img = self.to_pil(pred_np)
            images.append(img)
        
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        self.logger.info(f"GIF saved: {output_path}")
