import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_model import AutoCfdModel
from .loss import MseLoss
from .cfd_vae import CfdVaeLite # Import our VAE
from diffusers import UNet2DConditionModel, DDPMScheduler

class LatentDiffusionCfdModel(AutoCfdModel):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        loss_fn: MseLoss,
        n_case_params: int,
        vae_weights_path: str,
        image_size: int = 64,
        latent_dim: int = 4,
        noise_scheduler_timesteps: int = 1000,
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params

        # --- 1. Load the pre-trained VAE and freeze it ---
        self.vae = CfdVaeLite(in_chan=self.out_chan, out_chan=self.out_chan, latent_dim=latent_dim)
        self.vae.load_state_dict(torch.load(vae_weights_path, map_location="cpu"))
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # --- 2. Initialize the Diffusion U-Net ---
        self.unet = UNet2DConditionModel(
            sample_size=self.vae.latent_spatial_size,
            in_channels=latent_dim, 
            out_channels=latent_dim,
            cross_attention_dim=self.in_chan + self.n_case_params,
        )

        # Enable gradient checkpointing to save VRAM
        self.unet.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )

    def forward(
        self, inputs: Tensor, case_params: Tensor, label: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if label is None:
            raise ValueError("LDM requires a label for training.")

        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Step 1: Encode the clean target image into the latent space
        # We only need the mean of the latent distribution for training
        with torch.no_grad():
            # Call the encoder explicitly to get the latent distribution
            target_latents_dist = self.vae.vae.encode(label).latent_dist
            # ---------------
        target_latents = target_latents_dist.sample() * 4.5578
        # Step 2: Add noise to the latents
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inputs.shape[2], inputs.shape[3])
        conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
        conditioning_signal = conditioning_signal.view(batch_size, self.in_chan + self.n_case_params, -1)
        # Permute the dimensions to [Batch, SequenceLength, FeatureDimension]
        conditioning_signal = conditioning_signal.permute(0, 2, 1)

        noise_pred = self.unet(
            sample=noisy_latents, 
            timestep=timesteps, 
            encoder_hidden_states=conditioning_signal
        ).sample

        loss = F.mse_loss(noise_pred, noise)

        return {
            "preds": noise_pred,
            "loss": {"mse": loss, "nmse": loss / (torch.square(noise).mean() + 1e-8)}
        }

    @torch.no_grad()
    def generate(
        self, inputs: Tensor, case_params: Tensor, mask: Optional[Tensor] = None, num_inference_steps: int = 50
    ) -> Tensor:
        batch_size = inputs.shape[0]

        # Prepare conditioning signal
        case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inputs.shape[2], inputs.shape[3])
        conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
        conditioning_signal = conditioning_signal.view(batch_size, self.in_chan + self.n_case_params, -1)
        
        # Start with random noise in the latent space
        latents = torch.randn(
            (batch_size, self.vae.vae.latent_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            device=inputs.device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop in latent space
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(sample=latents, timestep=t, encoder_hidden_states=conditioning_signal).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the clean latents back to a full-resolution image
        latents = 1 / 4.5578 * latents # Unscale
        image = self.vae.vae.decode(latents).sample
        return image

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        if inputs.dim() == 3:
            inputs, case_params, mask = inputs.unsqueeze(0), case_params.unsqueeze(0), mask.unsqueeze(0)
            
        generated_frames = []
        current_frame = inputs
        print(f"Generating {steps} frames with Latent Diffusion...")
        for _ in tqdm(range(steps)):
            next_frame = self.generate(current_frame, case_params, mask)
            if mask is not None:
                next_frame = next_frame * mask
            generated_frames.append(next_frame)
            current_frame = next_frame
        return generated_frames
