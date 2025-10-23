"""
PUNetG-inspired U-Net for CFD Diffusion Model
Updated with accurate ResNetBlock structure from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        # Normalize over channel dimension
        norm = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        x_normed = x / norm
        # Scale by learnable weight
        return x_normed * self.weight[None, :, None, None]


class ResNetBlock(nn.Module):
    """
    ResNet block with timestep conditioning injection.
    Based on the paper's architecture diagram.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Timestep embedding projection (top branch in diagram)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Main feature path (bottom branch in diagram)
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = RMSNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # Skip connection (if channels change)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
            time_emb: [B, time_embed_dim] timestep + condition embedding
        Returns:
            [B, out_channels, H, W]
        """
        # Store for skip connection
        skip = self.skip(x)
        
        # Main path - first block
        # LayerNorm expects [B, H, W, C], so we need to permute
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        h = self.norm1(h)
        h = h.permute(0, 3, 1, 2)  # [B, C, H, W]
        h = self.act(h)
        h = self.conv1(h)  # [B, out_channels, H, W]
        
        # Add timestep conditioning (first + in diagram)
        time_scale = self.time_mlp(time_emb)  # [B, out_channels]
        time_scale = time_scale[:, :, None, None]  # [B, out_channels, 1, 1]
        h = h + time_scale
        
        # Main path - second block
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Add skip connection (second + in diagram)
        return h + skip


class Downsample(nn.Module):
    """Spatial downsampling by 2x."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by 2x."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer timesteps
        Returns:
            [B, dim] embeddings
        """
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PUNetGCFD(nn.Module):
    """
    PUNetG-style U-Net for CFD without attention mechanisms.
    Uses ResNetBlock structure from the paper diagram.
    """
    def __init__(
        self,
        in_channels: int = 2,           # u, v velocities
        out_channels: int = 2,
        base_channels: int = 64,        # C in the paper
        n_case_params: int = 5,         # Physical parameters
        channel_mults: tuple = (1, 2, 4),  # Channel multipliers per level
        num_res_blocks: int = 2,        # ResNet blocks per level
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Conditioning embeddings
        time_embed_dim = base_channels * 4
        
        # Timestep embedding (σ in diagram)
        self.time_embed = nn.Sequential(
            TimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Case parameters embedding (y in diagram)
        self.cond_embed = nn.Sequential(
            nn.Linear(n_case_params, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Combined conditioning dimension passed to ResNetBlocks
        # We'll concatenate time + case params
        combined_embed_dim = time_embed_dim * 2
        
        # Input projection (x_σ in diagram - geometry/velocity field)
        # Include mask as additional channel
        self.conv_in = nn.Conv2d(in_channels + 1, base_channels, 3, padding=1)
        
        # Build encoder blocks
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        channels = [base_channels * m for m in channel_mults]
        in_ch = base_channels
        
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResNetBlock(in_ch, out_ch, combined_embed_dim, dropout)
                )
                in_ch = out_ch
            self.down_blocks.append(blocks)
            
            # Downsample (except at last level)
            if i < len(channels) - 1:
                self.downsamplers.append(Downsample(out_ch))
            else:
                self.downsamplers.append(nn.Identity())
        
        # Bottleneck
        self.mid_block1 = ResNetBlock(
            channels[-1], channels[-1], combined_embed_dim, dropout
        )
        self.mid_block2 = ResNetBlock(
            channels[-1], channels[-1], combined_embed_dim, dropout
        )
        
        # Build decoder blocks
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        
        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1] if i < len(reversed_channels) - 1 else base_channels
            
            # Upsample (except at first decoder level)
            if i > 0:
                self.upsamplers.append(Upsample(in_ch))
            else:
                self.upsamplers.append(nn.Identity())
            
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # First block receives skip connection
                block_in = in_ch * 2 if j == 0 else out_ch
                blocks.append(
                    ResNetBlock(block_in, out_ch, combined_embed_dim, dropout)
                )
            
            self.up_blocks.append(blocks)
        
        # Output projection (ConvOut in diagram)
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,                    # [B, in_channels, H, W]
        timesteps: torch.Tensor,             # [B]
        case_params: torch.Tensor,           # [B, n_case_params]
        mask: Optional[torch.Tensor] = None, # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Forward pass matching the PUNetG architecture.
        
        Returns:
            [B, out_channels, H, W] predicted noise
        """
        # Add mask channel if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            x = torch.cat([x, mask], dim=1)
        else:
            # Add dummy mask channel (all ones)
            mask_dummy = torch.ones_like(x[:, :1])
            x = torch.cat([x, mask_dummy], dim=1)
        
        # Create combined conditioning embedding
        # This combines TimeEmbed (σ) and CondEmbed (y) from diagram
        t_emb = self.time_embed(timesteps)      # [B, time_embed_dim]
        c_emb = self.cond_embed(case_params)    # [B, time_embed_dim]
        cond_emb = torch.cat([t_emb, c_emb], dim=-1)  # [B, combined_embed_dim]
        
        # Input projection (ConvIn in diagram)
        h = self.conv_in(x)
        
        # Encoder path
        skip_connections = []
        for blocks, downsampler in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, cond_emb)
            skip_connections.append(h)
            h = downsampler(h)
        
        # Bottleneck
        h = self.mid_block1(h, cond_emb)
        h = self.mid_block2(h, cond_emb)
        
        # Decoder path (with skip connections marked by red arrows in diagram)
        for i, (upsampler, blocks) in enumerate(zip(self.upsamplers, self.up_blocks)):
            h = upsampler(h)
            
            # Add skip connection from encoder
            skip = skip_connections[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, cond_emb)
        
        # Output projection
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PUNetG CFD Model")
    print("=" * 60)
    
    model = PUNetGCFD(
        in_channels=2,
        out_channels=2,
        base_channels=64,
        n_case_params=5,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 2, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    case_params = torch.randn(batch_size, 5)
    mask = torch.ones(batch_size, 1, 64, 64)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")
    print(f"  Case params shape: {case_params.shape}")
    print(f"  Mask shape: {mask.shape}")
    
    output = model(x, timesteps, case_params, mask)
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Output shape should match input shape"
    print("\n✅ All tests passed!")
    
    # Memory estimate
    print(f"\n💾 Estimated memory per batch item:")
    print(f"  Forward pass: ~{(output.numel() * 4) / 1e6:.1f} MB")