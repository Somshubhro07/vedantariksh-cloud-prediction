# config/config.py
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import torch

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    in_channels: int = 3  # 2 conditioning frames + 1 noisy frame
    out_channels: int = 1
    base_channels: int = 64
    num_groups: int = 8  # For GroupNorm
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    use_attention: bool = True
    
@dataclass
class DiffusionConfig:
    """Diffusion process configuration"""
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "cosine"  # "linear" or "cosine"
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 1000
    save_every: int = 50
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    
@dataclass
class DataConfig:
    """Data configuration"""
    root_dir: str = "data/samples"
    input_frames: int = 2
    output_frames: int = 4
    image_size: Tuple[int, int] = (256, 256)
    augmentation: bool = True
    normalize: bool = True
    
@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # Device and paths
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
