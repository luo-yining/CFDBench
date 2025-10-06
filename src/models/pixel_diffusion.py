import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_model import AutoCfdModel
from .loss import MseLoss
from diffusers import UNet2DConditionModel, DDPMScheduler


class PixelDiffusionCfdModel(AutoCfdModel):
    """
    Pixel-space Diffusion Model using cross-attention conditioning.

    Similar to LatentDiffusionCfdModel but operates directly in pixel space
    without a VAE encoder/decoder.
    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        loss_fn: MseLoss,
        n_case_params: int,
        image_size: int = 64,
        noise_scheduler_timesteps: int = 1000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.image_size = image_size

        # Initialize the Diffusion U-Net operating in pixel space
        # Using architecture similar to CfdVaeLite
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=out_chan,  # Operating on pixel-space images
            out_channels=out_chan,
            cross_attention_dim=self.in_chan + self.n_case_params,
            # Architecture inspired by CfdVaeLite
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ),
            block_out_channels=(32, 64, 128, 256),
        )

        # Enable gradient checkpointing to save VRAM (optional, slows training)
        if use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )

    def forward(
        self, inputs: Tensor, label: Optional[Tensor] = None, case_params: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if label is None:
            raise ValueError("Pixel Diffusion requires a label for training.")

        batch_size = inputs.shape[0]
        device = inputs.device

        # Step 1: Add noise directly to the target image (pixel space)
        noise = torch.randn_like(label)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_images = self.noise_scheduler.add_noise(label, noise, timesteps)

        # Step 2: Prepare conditioning signal
        case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inputs.shape[2], inputs.shape[3])
        conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
        conditioning_signal = conditioning_signal.view(batch_size, self.in_chan + self.n_case_params, -1)
        # Permute the dimensions to [Batch, SequenceLength, FeatureDimension]
        conditioning_signal = conditioning_signal.permute(0, 2, 1)

        # Step 3: Predict noise
        noise_pred = self.unet(
            sample=noisy_images,
            timestep=timesteps,
            encoder_hidden_states=conditioning_signal
        ).sample

        # Step 4: Compute loss
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
        conditioning_signal = conditioning_signal.permute(0, 2, 1)

        # Start with random noise in pixel space
        images = torch.randn(
            (batch_size, self.out_chan, self.image_size, self.image_size),
            device=inputs.device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop in pixel space
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(sample=images, timestep=t, encoder_hidden_states=conditioning_signal).sample
            images = self.noise_scheduler.step(noise_pred, t, images).prev_sample

        return images

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        if inputs.dim() == 3:
            inputs, case_params, mask = inputs.unsqueeze(0), case_params.unsqueeze(0), mask.unsqueeze(0)

        generated_frames = []
        current_frame = inputs
        print(f"Generating {steps} frames with Pixel Diffusion...")
        for _ in tqdm(range(steps)):
            next_frame = self.generate(current_frame, case_params, mask)
            if mask is not None:
                next_frame = next_frame * mask
            generated_frames.append(next_frame)
            current_frame = next_frame
        return generated_frames
