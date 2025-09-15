import torch
from torch import nn
from diffusers import AutoencoderKL
from diffusers.configuration_utils import ConfigMixin
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

