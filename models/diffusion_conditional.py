import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class ConditionalDiffusion(nn.Module):
    """Enhanced conditional diffusion with cosine scheduling"""
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine"
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, conditioning_frames: torch.Tensor, t: torch.Tensor):
        # x_start: [batch, seq_len, channels, height, width] -> [batch, 2, 1, 6, 128, 128]
        # conditioning_frames: [batch, seq_len, channels, height, width] -> [batch, 4, 1, 6, 128, 128]
        
        batch_size = x_start.shape[0]
        
        # Reshape to [batch, seq_len * channels, height, width]
        # x_start: [batch, 2, 1, 6, 128, 128] -> [batch, 12, 128, 128]
        x_start_flat = x_start.view(batch_size, -1, x_start.shape[-2], x_start.shape[-1])
        
        # conditioning_frames: [batch, 4, 1, 6, 128, 128] -> [batch, 24, 128, 128]
        conditioning_flat = conditioning_frames.view(batch_size, -1, conditioning_frames.shape[-2], conditioning_frames.shape[-1])
        
        noise = torch.randn_like(x_start_flat)
        x_noisy = self.q_sample(x_start_flat, t, noise)
        
        # Concatenate: [batch, 24, 128, 128] + [batch, 12, 128, 128] = [batch, 36, 128, 128]
        model_input = torch.cat([conditioning_flat, x_noisy], dim=1)
        
        predicted_noise = self.model(model_input, t)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, conditioning_frames: torch.Tensor, t: torch.Tensor):
        # Handle the same reshaping as in p_losses
        batch_size = x.shape[0]
        conditioning_flat = conditioning_frames.view(batch_size, -1, conditioning_frames.shape[-2], conditioning_frames.shape[-1])
        
        model_input = torch.cat([conditioning_flat, x], dim=1)
        predicted_noise = self.model(model_input, t)
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        pred_mean = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        if t[0] == 0:
            return pred_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return pred_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, conditioning_frames: torch.Tensor, shape: tuple):
        device = conditioning_frames.device
        batch_size = conditioning_frames.shape[0]
        
        # shape should be for the flattened target: [12, 128, 128] for 2 frames * 6 channels
        x = torch.randn(batch_size, *shape[1:], device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, conditioning_frames, t)
        
        return x