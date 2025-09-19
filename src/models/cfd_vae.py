import torch
from torch import nn, Tensor
from typing import Dict

# Import from Hugging Face diffusers library
from diffusers import AutoencoderKL
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class CfdVae(ModelMixin, ConfigMixin):
    """
    A custom Variational Autoencoder for CFD data, built using the
    flexible AutoencoderKL class from Hugging Face diffusers.
    """
    def __init__(self, in_chan=2, out_chan=2, latent_dim=4):
        super().__init__()
        
        # Define a custom configuration for our VAE.
        # This architecture is simpler and tailored for our 2-channel data.
        self.vae = AutoencoderKL(
            in_channels=in_chan,
            out_channels=out_chan,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128), # Channel configuration at each stage
            latent_channels=latent_dim,
            sample_size=64
        )

    def forward(self, x):
        """
        The forward pass of the VAE.
        Takes a batch of images and returns the reconstructed images
        and the latent distribution.
        """
        return self.vae(x)



class CfdVae2(ModelMixin, ConfigMixin):
    """
    A custom Variational Autoencoder for CFD data, built using the
    flexible AutoencoderKL class from Hugging Face diffusers.
    
    This version has 3 downsampling blocks for higher compression.
    """
    def __init__(self, in_chan=2, out_chan=2, latent_dim=4):
        super().__init__()
        
        # Define a custom configuration for our VAE with 3 down/up blocks
        # to achieve an 8x spatial downsampling (64x64 -> 8x8).
        self.vae = AutoencoderKL(
            in_channels=in_chan,
            out_channels=out_chan,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128, 256), # Channel configuration at each stage
            latent_channels=latent_dim,
            sample_size=64
        )

    def forward(self, x):
        """
        The forward pass of the VAE.
        Takes a batch of images and returns the reconstructed images
        and the latent distribution.
        """
        return self.vae(x)


class CfdVae3(ModelMixin, ConfigMixin):
    """
    A wrapper class for a custom-configured Variational Autoencoder (VAE) 
    from the Hugging Face diffusers library, tailored for CFD data.
    
    This VAE is designed to compress a 64x64 CFD frame into a much smaller
    8x8 latent space.
    """
    @register_to_config
    def __init__(self, in_chan: int, out_chan: int, latent_dim: int = 4):
        super().__init__()
        
        # Store key dimensions for easy access later
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.latent_channels = latent_dim
        # This is the intended spatial size after 3 downsampling blocks (64 -> 32 -> 16 -> 8)
        self.latent_spatial_size = 8 

        # --- THE FINAL FIX ---
        # To get 3 downsampling steps, we must provide a list of 4 blocks.
        # The last block does not perform downsampling.
        self.vae = AutoencoderKL(
            in_channels=in_chan,
            out_channels=out_chan,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128, 256, 512), # Added a channel dimension for the 4th block
            latent_channels=latent_dim,
            sample_size=64,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        The forward method for the VAE. It takes a batch of images, encodes them,
        samples from the latent distribution, and decodes them back into images.
        """
        return self.vae(x)