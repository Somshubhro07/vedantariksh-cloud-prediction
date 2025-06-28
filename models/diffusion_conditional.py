# models/diffusion_conditional.py
from regex import F
import torch
import torch.nn as nn


class ConditionalDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.alpha_cumprod[t] ** 0.5
        sqrt_one_minus = (1 - self.alpha_cumprod[t]) ** 0.5
        return sqrt_alpha.view(-1, 1, 1, 1) * x_start + sqrt_one_minus.view(-1, 1, 1, 1) * noise

    def p_losses(self, x_start, conditioning_frames, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Concatenate [conditioning_frames, x_noisy] along channel dim
        model_input = torch.cat([conditioning_frames, x_noisy], dim=1)
        predicted_noise = self.model(model_input)
        return F.mse_loss(predicted_noise, noise)
