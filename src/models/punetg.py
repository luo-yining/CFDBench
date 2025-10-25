"""
PUNetG-inspired U-Net for CFD Diffusion Model
Based on user-provided code, adapted slightly for clarity and structure.
Includes ResNetBlock structure inspired by diffusion model papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List # Added List for skip_connections type hint


# RMSNorm remains here, although not currently used by ResNetBlock below
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate RMS over the channel dimension (dim=1)
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        """
        output = self._norm(x.float()).type_as(x)
        # Apply learnable weight, broadcasting along H and W dimensions
        return output * self.weight[None, :, None, None]


class ResNetBlock(nn.Module):
    """
    ResNet block with timestep and conditioning embedding injection.
    Uses GroupNorm and SiLU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # Combined dimension of timestep + case_param embeddings
        condition_embed_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32, # Added num_groups parameter
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Conditioning Projection ---
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_embed_dim, out_channels * 2), # Project to get scale and shift
        )

        # --- Main Feature Path ---
        # Ensure num_groups is not larger than in_channels
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Ensure num_groups is not larger than out_channels
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # --- Skip Connection ---
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, condition_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        cond_proj = self.condition_mlp(condition_emb)[:, :, None, None]
        scale, shift = cond_proj.chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale) + shift # Modulate features
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class Downsample(nn.Module):
    """Spatial downsampling layer using strided convolution."""
    def __init__(self, channels: int):
        super().__init__()
        # Use Conv2d with stride 2 for downsampling
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling layer using interpolation and convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding module."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"Timesteps must be a 1D tensor, got shape {timesteps.shape}")

        half_dim = self.dim // 2
        exponent = -torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class PUNetGCFD(nn.Module):
    """
    PUNetG-style U-Net for CFD, adapted for diffusion model conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_case_params: int = 5,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_groups_norm: int = 32, # Number of groups for GroupNorm
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- Conditioning Embeddings ---
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(n_case_params, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        combined_embed_dim = time_embed_dim * 2

        # --- U-Net Architecture ---
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResNetBlock(current_channels, out_ch, combined_embed_dim, dropout, num_groups=num_groups_norm)
                )
                current_channels = out_ch
                channels.append(current_channels) # Store channels after each ResBlock
            is_last = (i == len(channel_mults) - 1)
            # Add downsampling unless it's the last level
            level_blocks.append(Downsample(current_channels) if not is_last else nn.Identity())
            self.down_blocks.append(level_blocks)


        # --- Bottleneck ---
        self.mid_block1 = ResNetBlock(
            current_channels, current_channels, combined_embed_dim, dropout, num_groups=num_groups_norm
        )
        self.mid_block2 = ResNetBlock(
            current_channels, current_channels, combined_embed_dim, dropout, num_groups=num_groups_norm
        )

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        # Iterate through channel multipliers in reverse
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult # Target output channels for this level
            level_blocks = nn.ModuleList()

            # Add Upsampling layer first (unless it's the first decoder level)
            is_first = (i == 0)
            level_blocks.append(Upsample(current_channels) if not is_first else nn.Identity())
            # After upsampling, current_channels matches the target out_ch for this level

            for j in range(num_res_blocks + 1): # +1 for the skip connection block
                # Get channel count from corresponding encoder skip connection
                skip_channels = channels.pop()
                block_in_channels = current_channels + skip_channels

                level_blocks.append(
                    ResNetBlock(
                        block_in_channels,
                        out_ch, # Output channels for ResBlock
                        combined_embed_dim,
                        dropout,
                        num_groups=num_groups_norm
                    )
                )
                current_channels = out_ch # Update current channels for next block/level

            self.up_blocks.append(level_blocks)


        # --- Output Convolution ---
        self.norm_out = nn.GroupNorm(min(num_groups_norm, base_channels), base_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        case_params: torch.Tensor,
        mask: Optional[torch.Tensor] = None # Mask is handled by GenCastCfdModel concatenating it
    ) -> torch.Tensor:
        """ Forward pass. """
        # --- 1. Prepare Conditioning Embedding ---
        t_emb = self.time_embed(timesteps)
        c_emb = self.cond_embed(case_params)
        cond_emb = torch.cat([t_emb, c_emb], dim=-1)

        # --- 2. Input Convolution ---
        h = self.conv_in(x)
        skip_connections: List[torch.Tensor] = []

        # --- 3. Encoder ---
        for level_blocks in self.down_blocks:
            # Process ResBlocks and store outputs for skip connections
            for block in level_blocks[:-1]: # All ResBlocks in the level
                 h = block(h, cond_emb)
                 skip_connections.append(h)
            # Apply Downsample (or Identity)
            downsampler = level_blocks[-1]
            h = downsampler(h)


        # --- 4. Bottleneck ---
        h = self.mid_block1(h, cond_emb)
        h = self.mid_block2(h, cond_emb)

        # --- 5. Decoder ---
        for level_blocks in self.up_blocks:
            # Apply Upsample (or Identity)
            upsampler = level_blocks[0]
            h = upsampler(h)

            # Process ResBlocks with skip connections
            for block in level_blocks[1:]: # All ResBlocks in the level
                 # Get skip connection from the end of the list
                 skip = skip_connections.pop()
                 # Concatenate skip connection before passing to ResBlock
                 h = torch.cat([h, skip], dim=1)
                 h = block(h, cond_emb)


        # --- 6. Output Convolution ---
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# Example usage and testing (optional, can be removed)
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PUNetG CFD Model (with fixes)")
    print("=" * 60)

    # Example: Match GenCastCfdModel input size
    # noisy_res(2) + x_t-1(2) + x_t-2(2) = 6 input channels
    model = PUNetGCFD(
        in_channels=6,
        out_channels=2, # Predicts noise for u, v
        base_channels=64,
        n_case_params=5,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
    ).cuda() # Move to GPU if available

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    print(f"\n🧪 Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 6, 64, 64).cuda() # Example input
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()
    case_params = torch.randn(batch_size, 5).cuda()
    # Mask is included in x's channels by the wrapper model

    print(f"  Input shape: {x.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")
    print(f"  Case params shape: {case_params.shape}")

    # Test with autocast for mixed precision compatibility
    try:
        from torch.amp import autocast
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(x, timesteps, case_params)
        print("  Forward pass with AMP successful.")
    except ImportError:
        output = model(x, timesteps, case_params)
        print("  Forward pass without AMP successful.")


    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 2, 64, 64), "Output shape mismatch!"
    print("\n✅ Forward pass shape test passed!")
