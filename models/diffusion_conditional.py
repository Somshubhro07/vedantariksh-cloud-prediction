# models/diffusion_conditional.py
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
        self.register_buffer('alphas_cumprod_prev', F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine beta schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, conditioning_frames: torch.Tensor, t: torch.Tensor):
        """Calculate training loss"""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Concatenate conditioning frames with noisy frame
        model_input = torch.cat([conditioning_frames, x_noisy], dim=1)
        predicted_noise = self.model(model_input, t)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, conditioning_frames: torch.Tensor, t: torch.Tensor):
        """Sample from p(x_{t-1} | x_t)"""
        model_input = torch.cat([conditioning_frames, x], dim=1)
        predicted_noise = self.model(model_input, t)
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute mean
        pred_mean = (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        if t[0] == 0:
            return pred_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return pred_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, conditioning_frames: torch.Tensor, shape: tuple):
        """Generate samples"""
        device = conditioning_frames.device
        batch_size = conditioning_frames.shape[0]
        
        # Start from random noise
        x = torch.randn(batch_size, *shape[1:], device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, conditioning_frames, t)
        
        return x
