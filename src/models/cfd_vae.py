import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

class CfdVae(nn.Module):
    """
    A Variational Autoencoder for CFD data, using the Hugging Face AutoencoderKL model.
    """
    def __init__(self, in_chan: int = 2, out_chan: int = 2, latent_dim: int = 4):
        super().__init__()
        # Using a standard architecture similar to the one used in Stable Diffusion
        self.vae = AutoencoderKL(
            in_channels=in_chan,
            out_channels=out_chan,
            latent_channels=latent_dim,
            block_out_channels=(64, 128), # Reduced complexity for smaller images
            layers_per_block=2,
            norm_num_groups=32
        )

    def forward(self, x: torch.Tensor):
        # The forward pass returns the full output dictionary
        return self.vae(x)
