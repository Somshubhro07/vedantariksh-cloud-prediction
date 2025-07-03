import os
from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Based on your training logs: 24 conditioning + 6 target = 30 input channels
    in_channels: int = 30  # Conditioning (24) + Target (6) 
    out_channels: int = 6  # 6 channels output (matches your logs)
    base_channels: int = 64
    num_groups: int = 8
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    use_attention: bool = True

@dataclass
class DiffusionConfig:
    """Diffusion process configuration"""
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "cosine"

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 1
    learning_rate: float = 1e-4
    epochs: int = 50  # Reduced from 500 for faster testing
    save_every: int = 5
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    
    # New parameters for better control
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    min_lr: float = 1e-6

@dataclass
class DataConfig:
    """Data configuration"""
    root_dir: str = "data/insat_data"
    input_frames: int = 4
    output_frames: int = 2
    image_size: Tuple[int, int] = (128, 128)
    augmentation: bool = True
    normalize: bool = True
    
    # New parameters for performance
    num_workers: int = 4  # For faster data loading
    pin_memory: bool = False  # Set to False for CPU training
    prefetch_factor: int = 2

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_mixed_precision: bool = False  # Set to True if using GPU
    
    # Directory configuration
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    # New: Performance and debugging options
    profile_training: bool = False
    debug_mode: bool = False
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Auto-adjust settings based on device
        if self.device == "cuda":
            self.data.pin_memory = True
            self.use_mixed_precision = True
            self.training.batch_size = min(self.training.batch_size * 2, 4)  # Can handle larger batches
        else:
            self.data.pin_memory = False
            self.use_mixed_precision = False
            self.data.num_workers = min(self.data.num_workers, 2)  # Reduce workers for CPU
            
        # Validate channel dimensions
        expected_conditioning = self.data.input_frames * 6  # 4 frames × 6 channels = 24
        expected_target = 6  # Single timestep of 6 channels for diffusion
        expected_input = expected_conditioning + expected_target  # 24 + 6 = 30
        
        if self.model.in_channels != expected_input:
            print(f"⚠️  Warning: Model input channels ({self.model.in_channels}) != expected ({expected_input})")
            print(f"   Auto-correcting to {expected_input}")
            self.model.in_channels = expected_input
            
        if self.model.out_channels != 6:
            print(f"⚠️  Warning: Model output channels ({self.model.out_channels}) != expected (6)")
            print(f"   Auto-correcting to 6")
            self.model.out_channels = 6

# Quick configuration presets
@dataclass
class FastConfig(Config):
    """Fast training configuration for testing"""
    def __post_init__(self):
        super().__post_init__()
        self.training.epochs = 10
        self.data.image_size = (64, 64)  # Smaller images
        self.training.batch_size = 2 if self.device == "cuda" else 1

@dataclass
class ProductionConfig(Config):
    """Production training configuration"""
    def __post_init__(self):
        super().__post_init__()
        self.training.epochs = 200
        self.training.batch_size = 4 if self.device == "cuda" else 1
        self.training.save_every = 10