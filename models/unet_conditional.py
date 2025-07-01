# models/unet_conditional.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from .attention import SelfAttention

class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and optional attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        num_groups: int = 8,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        # Attention
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
    
    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        h = h + self.skip(x)
        
        # Attention
        if self.attention is not None:
            h = self.attention(h)
        
        return h

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalUNet(nn.Module):
    """Enhanced UNet with time embedding and attention"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        num_groups: int = 8,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_emb_dim, num_groups, dropout),
            ResidualBlock(base_channels, base_channels, time_emb_dim, num_groups, dropout),
        ])
        self.down_sample1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        
        self.mid_blocks1 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, num_groups, dropout),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, num_groups, dropout, use_attention),
        ])
        self.down_sample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, num_groups, dropout, use_attention),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, num_groups, dropout, use_attention),
        ])
        
        # Decoder
        self.up_sample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_blocks1 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, num_groups, dropout, use_attention),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, num_groups, dropout),
        ])
        
        self.up_sample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_blocks2 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim, num_groups, dropout),
            ResidualBlock(base_channels, base_channels, time_emb_dim, num_groups, dropout),
        ])
        
        # Output
        self.out_norm = nn.GroupNorm(num_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, timesteps=None):
        # Time embedding
        if timesteps is not None:
            time_emb = self.time_embed(timesteps)
        else:
            time_emb = None
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for block in self.down_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
        h = self.down_sample1(h)
        
        for block in self.mid_blocks1:
            h = block(h, time_emb)
            skip_connections.append(h)
        h = self.down_sample2(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, time_emb)
        
        # Decoder
        h = self.up_sample1(h)
        h = torch.cat([h, skip_connections.pop()], dim=1)
        for block in self.up_blocks1:
            h = block(h, time_emb)
        
        h = self.up_sample2(h)
        h = torch.cat([h, skip_connections.pop()], dim=1)
        for block in self.up_blocks2:
            h = block(h, time_emb)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
