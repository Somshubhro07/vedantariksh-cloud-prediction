import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class SelfAttention(nn.Module):
    """Self-attention module for spatial attention"""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels {channels} not divisible by num_heads {num_heads}"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).reshape(B, C, H, W)  # Use reshape instead of view
        
        # Output projection
        out = self.out(out)
        
        # Residual connection
        return x + out

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
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()
        
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
    
    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.attention is not None:
            h = self.attention(h)
        
        return h + self.skip(x)

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
        in_channels: int = 30,  # 24 (conditioning) + 6 (target) = 30
        out_channels: int = 6,  # 6 channels per frame
        base_channels: int = 64,
        num_groups: int = 8,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        time_emb_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList([
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, num_groups, dropout),
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, num_groups, dropout),
        ])
        self.down_sample1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, num_groups, dropout),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, num_groups, dropout, use_attention),
        ])
        self.down_sample2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        
        self.bottleneck = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, num_groups, dropout, use_attention),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, num_groups, dropout, use_attention),
        ])
        
        # Decoder - Fixed channel calculations
        self.up_sample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1, output_padding=0)
        self.up_blocks1 = nn.ModuleList([
            # Input: concat(upsampled: 128, skip: 256) = 384 channels
            ResidualBlock(base_channels * 6, base_channels * 2, time_emb_dim, num_groups, dropout, use_attention),  # 384 -> 128
            ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, num_groups, dropout),  # 128 -> 128
        ])
        
        self.up_sample2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1, output_padding=0)
        self.up_blocks2 = nn.ModuleList([
            # Input: concat(upsampled: 64, skip: 128) = 192 channels
            ResidualBlock(base_channels * 3, base_channels, time_emb_dim, num_groups, dropout),  # 192 -> 64
            ResidualBlock(base_channels, base_channels, time_emb_dim, num_groups, dropout),  # 64 -> 64
        ])
        
        self.out_norm = nn.GroupNorm(num_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, timesteps=None):
        if timesteps is not None:
            time_emb = self.time_embed(timesteps)
        else:
            time_emb = None
        
        h = self.init_conv(x)  # [batch, 64, 128, 128]
        
        # Encoder
        skip_connections = []
        h = self.down_blocks[0](h, time_emb)  # [batch, 128, 128, 128]
        h = self.down_blocks[1](h, time_emb)  # [batch, 128, 128, 128]
        skip_connections.append(h)
        h = self.down_sample1(h)  # [batch, 128, 64, 64]
        h = self.mid_blocks[0](h, time_emb)  # [batch, 256, 64, 64]
        h = self.mid_blocks[1](h, time_emb)  # [batch, 256, 64, 64]
        skip_connections.append(h)
        h = self.down_sample2(h)  # [batch, 256, 32, 32]
        for block in self.bottleneck:
            h = block(h, time_emb)
        
        # Decoder
        h = self.up_sample1(h)  # [batch, 128, 64, 64]
        skip = skip_connections.pop()  # [batch, 256, 64, 64]
        h = torch.cat([h, skip], dim=1)  # [batch, 384, 64, 64]
        h = self.up_blocks1[0](h, time_emb)  # [batch, 128, 64, 64]
        h = self.up_blocks1[1](h, time_emb)  # [batch, 128, 64, 64]
        h = self.up_sample2(h)  # [batch, 64, 128, 128]
        skip = skip_connections.pop()  # [batch, 128, 128, 128]
        h = torch.cat([h, skip], dim=1)  # [batch, 192, 128, 128]
        h = self.up_blocks2[0](h, time_emb)  # [batch, 64, 128, 128]
        h = self.up_blocks2[1](h, time_emb)  # [batch, 64, 128, 128]
        
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h