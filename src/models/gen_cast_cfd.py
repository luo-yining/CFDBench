import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

# CFDBench Imports (adjust path if needed)
from .base_model import AutoCfdModel
from .loss import MseLoss
# Assuming PUNetGCFD is in the same directory or properly added to PYTHONPATH
# If it's elsewhere, adjust the import, e.g., from ..my_models import PUNetGCFD
try:
    from .punetg import PUNetGCFD
except ImportError:
    print("Warning: PUNetGCFD not found. Please ensure it's defined in src/models/punetg.py or adjust the import.")
    # Define a placeholder if PUNetG is missing, to allow code structure check
    class PUNetGCFD(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Using Placeholder PUNetGCFD!")
            self.layer = nn.Conv2d(kwargs.get('in_channels', 4), kwargs.get('out_channels', 2), kernel_size=1)
        def forward(self, x, timesteps, case_params, mask):
            # Basic passthrough for placeholder
            return self.layer(x[:, :self.layer.out_channels]) # Return noise shaped output


# Diffusers Import
try:
    from diffusers import DDPMScheduler
except ImportError:
    print("Warning: diffusers library not found. Please install it (`pip install diffusers transformers accelerate`)")
    # Define a placeholder if diffusers is missing
    class DDPMScheduler:
         def __init__(self, *args, **kwargs): self.config = type('obj', (object,), {'num_train_timesteps': 1000})()
         def add_noise(self, x, noise, ts): return x + noise
         def set_timesteps(self, n): self.timesteps = torch.arange(n)
         def step(self, noise_pred, t, latents): return type('obj', (object,), {'prev_sample': latents - noise_pred})()


class GenCastCfdModel(AutoCfdModel):
    """
    Conditional, autoregressive Pixel-space Diffusion Model inspired by GenCast.

    Predicts the *normalized residual* between frames (X_t - X_{t-1}),
    using second-order conditioning (X_{t-1} and X_{t-2}).

    Conditioning Signals:
    1. X_{t-1} (inputs): Spatial conditioning via channel concatenation.
    2. X_{t-2} (inputs_prev): Spatial conditioning via channel concatenation.
    3. Case parameters (case_params): Global conditioning via PUNetG embedding.
    4. Diffusion Timestep (t): Global conditioning via PUNetG embedding.
    5. Mask (mask): Handled internally by PUNetG.
    """
    def __init__(
        self,
        in_chan: int,        # Channels of the input frame (e.g., 2 for u, v)
        out_chan: int,       # Channels of the output/residual (e.g., 2 for u, v)
        loss_fn: MseLoss,    # Loss function instance (e.g., MseLoss(normalize=True))
        n_case_params: int,  # Number of case parameters
        residual_mean: torch.Tensor, # Pre-calculated mean of residuals
        residual_std: torch.Tensor,  # Pre-calculated std dev of residuals
        image_size: int = 64,        # Height/Width of the input frames
        noise_scheduler_timesteps: int = 1000, # Number of diffusion steps
        use_gradient_checkpointing: bool = True, # Use checkpointing in PUNetG if available
        # PUNetG Specific Args (passed via **kwargs or directly)
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(loss_fn) # Pass loss_fn to the base class
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.image_size = image_size

        # --- U-Net Input Channels Calculation ---
        # Input: noisy normalized residual (out_chan)
        # Condition 1: X_{t-1} (in_chan)
        # Condition 2: X_{t-2} (in_chan)
        unet_in_channels = out_chan + in_chan + in_chan

        # --- Initialize PUNetG U-Net ---
        # PUNetG should handle timestep and case_param conditioning internally
        self.unet = PUNetGCFD(
            in_channels=unet_in_channels,
            out_channels=out_chan,       # Output predicts noise (same channels as residual)
            base_channels=base_channels,
            n_case_params=n_case_params, # Pass n_case_params for embedding
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            # Add other PUNetG specific args if needed
        )

        # Gradient checkpointing control
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Note: If PUNetGCFD doesn't have a built-in method like enable_gradient_checkpointing(),
        # you'll need to wrap its forward call manually using torch.utils.checkpoint.checkpoint
        # (as shown in the forward method).

        # --- Noise Scheduler ---
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_timesteps,
            beta_schedule="squaredcos_cap_v2" # A common schedule, adjust if needed
        )

        # --- Store Residual Normalization Statistics ---
        # Register as buffers: ensures they are part of the model's state_dict,
        # moved to the correct device (e.g., GPU), but are not trained.
        # Ensure shape [1, C, 1, 1] for broadcasting during normalization.
        if residual_mean.ndim == 1:
            residual_mean = residual_mean.view(1, -1, 1, 1)
        if residual_std.ndim == 1:
            residual_std = residual_std.view(1, -1, 1, 1)
        self.register_buffer('residual_mean', residual_mean)
        self.register_buffer('residual_std', residual_std)


    def forward(
        self,
        inputs: Tensor,             # X_{t-1} velocity [B, in_chan, H, W]
        inputs_prev: Tensor,        # X_{t-2} velocity [B, in_chan, H, W]
        label: Optional[Tensor],    # X_{t} velocity   [B, out_chan, H, W]
        case_params: Tensor,        # Case parameters  [B, n_case_params]
        mask: Optional[Tensor],     # Mask             [B, 1, H, W]
        **kwargs,                   # Allow extra unused args from dataloader
    ) -> Dict[str, Tensor]:
        """
        Training forward pass. Calculates the diffusion loss.
        """
        if label is None:
            raise ValueError("GenCastCfdModel requires a 'label' during training.")
        if mask is None:
             print("Warning: Mask not provided to forward pass. Assuming all regions are valid.")
             mask = torch.ones_like(label[:, 0:1]) # Create dummy mask if needed

        batch_size = inputs.shape[0]
        device = inputs.device

        # --- 1. Calculate and Normalize the Residual ---
        raw_residual = label - inputs
        # Normalize: (X_t - X_{t-1} - mean) / std
        # Add epsilon to std to prevent division by zero
        normalized_residual = (raw_residual - self.residual_mean) / (self.residual_std + 1e-6)

        # --- 2. Sample Noise and Timesteps ---
        noise = torch.randn_like(normalized_residual)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # --- 3. Add Noise to Normalized Residual ---
        noisy_normalized_residual = self.noise_scheduler.add_noise(
            original_samples=normalized_residual,
            noise=noise,
            timesteps=timesteps
        )

        # --- 4. Prepare U-Net Input: Concatenate Conditions ---
        # [noisy_residual (out), X_{t-1} (in), X_{t-2} (in)]
        unet_input = torch.cat([noisy_normalized_residual, inputs, inputs_prev], dim=1)

        # --- 5. Predict Noise using PUNetG ---
        # PUNetG expects: (x, timesteps, case_params, mask)
        if self.use_gradient_checkpointing and self.training:
            # Manual checkpointing if PUNetG doesn't support it internally
             noise_pred = torch.utils.checkpoint.checkpoint(
                 self.unet,
                 unet_input,
                 timesteps,
                 case_params,
                 mask,
                 use_reentrant=False # Recommended for newer PyTorch versions
             )
        else:
             noise_pred = self.unet(
                 x=unet_input,
                 timesteps=timesteps,
                 case_params=case_params,
                 mask=mask # Pass mask to PUNetG
             )

        # --- 6. Compute Loss against the original noise ---
        # The loss_fn (e.g., MseLoss) compares the predicted noise to the actual noise added.
        # Ensure loss is calculated only on valid regions using the mask if loss_fn supports it.
        # If loss_fn doesn't support masking, apply mask manually:
        # loss_dict = self.loss_fn(noise_pred * mask, noise * mask)
        # Or, if MseLoss handles it internally based on its init flag:
        loss_dict = self.loss_fn(preds=noise_pred, labels=noise)

        # Make sure 'mse' is the primary loss for optimization
        if 'mse' not in loss_dict:
             # If MseLoss returns nmse primarily, calculate mse
             loss_dict['mse'] = loss_dict.get('nmse', torch.tensor(0.0, device=device)) * (torch.square(noise).mean() + 1e-8)


        return {
            "preds": noise_pred, # Return noise prediction
            "loss": loss_dict    # Return dict of losses (e.g., {'mse': ..., 'nmse': ...})
        }

    @torch.no_grad()
    def generate(
        self,
        inputs: Tensor,         # X_{t-1} velocity [B, in_chan, H, W]
        inputs_prev: Tensor,    # X_{t-2} velocity [B, in_chan, H, W]
        case_params: Tensor,    # Case parameters  [B, n_case_params]
        mask: Optional[Tensor], # Mask             [B, 1, H, W]
        num_inference_steps: int = 50, # Number of DDPM steps
        **kwargs,               # Allow extra unused args
    ) -> Tensor:
        """
        Generates the *next* frame (X_t) given the previous two frames (X_{t-1}, X_{t-2}).
        Performs the full DDPM sampling process to predict the residual,
        then adds it to the input X_{t-1}.
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        if mask is None:
             mask = torch.ones_like(inputs[:, 0:1]) # Dummy mask if not provided

        # --- 1. Start with Random Noise in Residual Shape ---
        # Shape: [B, out_chan, H, W]
        residual_shape = (batch_size, self.out_chan, self.image_size, self.image_size)
        # This tensor will be iteratively denoised to become the *normalized* residual
        pred_norm_residual = torch.randn(residual_shape, device=device)

        # --- 2. Set Denoising Timesteps ---
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # --- 3. Denoising Loop ---
        for t in tqdm(self.noise_scheduler.timesteps, desc="DDPM Sampling", leave=False):
            # Expand t to batch size
            timestep_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # --- Prepare U-Net Input: Concatenate Conditions ---
            # [current_noisy_residual (out), X_{t-1} (in), X_{t-2} (in)]
            unet_input = torch.cat([pred_norm_residual, inputs, inputs_prev], dim=1)

            # --- Predict Noise ---
            noise_pred = self.unet(
                x=unet_input,
                timesteps=timestep_batch,
                case_params=case_params,
                mask=mask
            )

            # --- Denoise One Step using Scheduler ---
            # The scheduler computes the previous sample (less noisy)
            pred_norm_residual = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=pred_norm_residual
            ).prev_sample # .prev_sample contains the denoised sample x_{t-1}

        # --- 4. De-normalize the Final Predicted Residual ---
        # pred_norm_residual is now the model's estimate of the clean normalized residual Z_t
        pred_residual = (pred_norm_residual * self.residual_std) + self.residual_mean

        # --- 5. Calculate the Next Frame ---
        # X_t = X_{t-1} + (X_t - X_{t-1})
        next_frame = inputs + pred_residual

        # --- 6. Apply Mask (Optional but Recommended) ---
        # Ensure the generated frame respects boundaries/obstacles defined by the mask
        if mask is not None:
             next_frame = next_frame * mask

        return next_frame

    # --- generate_many remains the same as previously provided ---
    # It correctly uses the updated generate() method for autoregression.
    def generate_many(
        self,
        inputs: Tensor,         # Initial frame X_t     [B, in_chan, H, W]
        inputs_prev: Tensor,    # Initial frame X_{t-1} [B, in_chan, H, W]
        case_params: Tensor,    # Case parameters       [B, n_params]
        mask: Tensor,           # Mask                  [B, 1, H, W]
        steps: int              # Number of steps to generate
    ) -> List[Tensor]:
        """ Autoregressively generates a sequence of 'steps' frames. """
        if inputs.dim() != 4 or inputs_prev.dim() != 4:
             raise ValueError("generate_many expects 4D input tensors (B, C, H, W)")

        generated_frames = []
        current_frame = inputs      # This is X_t for the first prediction
        prev_frame = inputs_prev    # This is X_{t-1} for the first prediction

        print(f"Generating {steps} frames autoregressively (GenCast-style)...")
        for _ in tqdm(range(steps), desc="Autoregressive Generation"):
            # Predict X_{t+1} using (X_t = current_frame) and (X_{t-1} = prev_frame)
            next_frame = self.generate(
                inputs=current_frame,
                inputs_prev=prev_frame,
                case_params=case_params,
                mask=mask
                # num_inference_steps can be passed via args if needed
            )

            generated_frames.append(next_frame)

            # Update the state for the next loop: t becomes t-1, t+1 becomes t
            prev_frame = current_frame
            current_frame = next_frame

        return generated_frames
