import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_model import AutoCfdModel
from .loss import MseLoss
from .punetg import PUNetGCFD
from diffusers import DDPMScheduler


class PixelDiffusionCfdModel(AutoCfdModel):
    """
    Pixel-space Diffusion Model using PUNetG architecture.

    Similar to LatentDiffusionCfdModel but operates directly in pixel space
    without a VAE encoder/decoder. Uses PUNetG-inspired U-Net with
    timestep and case parameter conditioning.
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
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.image_size = image_size

        # Initialize PUNetG U-Net operating in pixel space
        # Note: PUNetG handles conditioning via timestep + case_params embeddings
        # rather than cross-attention
        self.unet = PUNetGCFD(
            in_channels=out_chan,  # Operating on pixel-space images (noisy target)
            out_channels=out_chan,
            base_channels=base_channels,
            n_case_params=n_case_params,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        # Note: PUNetG doesn't have gradient checkpointing built-in like HF models
        # If needed, it can be added by wrapping forward pass with torch.utils.checkpoint
        self.use_gradient_checkpointing = use_gradient_checkpointing

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

        # Step 2: Predict noise using PUNetG
        # PUNetG takes: (x, timesteps, case_params, mask)
        # Note: PUNetG handles mask internally by concatenating it with input
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            noise_pred = torch.utils.checkpoint.checkpoint(
                self.unet,
                noisy_images,
                timesteps,
                case_params,
                mask,
                use_reentrant=False
            )
        else:
            noise_pred = self.unet(
                x=noisy_images,
                timesteps=timesteps,
                case_params=case_params,
                mask=mask
            )

        # Step 3: Compute loss using the loss function to get all metrics
        # This ensures compatibility with evaluation code that expects all score_names
        loss_dict = self.loss_fn(noise_pred, noise)

        return {
            "preds": noise_pred,
            "loss": loss_dict
        }

    @torch.no_grad()
    def generate(
        self, inputs: Tensor, case_params: Tensor, mask: Optional[Tensor] = None, num_inference_steps: int = 50
    ) -> Tensor:
        batch_size = inputs.shape[0]

        # Start with random noise in pixel space
        images = torch.randn(
            (batch_size, self.out_chan, self.image_size, self.image_size),
            device=inputs.device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop in pixel space using PUNetG
        for t in self.noise_scheduler.timesteps:
            # Create timestep tensor for the batch
            timestep_batch = torch.full((batch_size,), t, device=inputs.device, dtype=torch.long)

            # Predict noise using PUNetG
            noise_pred = self.unet(
                x=images,
                timesteps=timestep_batch,
                case_params=case_params,
                mask=mask
            )

            # Denoise one step
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
