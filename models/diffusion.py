# models/diffusion.py

import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alpha_cumprod[t] ** 0.5
        sqrt_one_minus = (1 - self.alpha_cumprod[t]) ** 0.5
        return sqrt_alpha_cumprod.view(-1, 1, 1, 1) * x_start + sqrt_one_minus.view(-1, 1, 1, 1) * noise

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy)
        return nn.MSELoss()(predicted_noise, noise)
