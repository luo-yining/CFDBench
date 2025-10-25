import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_model import AutoCfdModel
from .loss import MseLoss
from .cfd_vae import CfdVaeLite # Import our VAE
from diffusers import UNet2DConditionModel, DDPMScheduler, UNet2DModel

class LatentDiffusionCfdModel2(AutoCfdModel):
    """
    Latent Diffusion Model using cross-attention conditioning.

    Uses a pre-trained VAE to compress flow fields to latent space, then
    trains a diffusion model with cross-attention conditioning on velocity
    fields and case parameters.
    """
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
        scaling_factor: float = 4.5578,
        # NEW: Memory-efficient U-Net parameters
        unet_base_channels: int = 64,  # Reduced from default 128/256
        unet_channel_mult: tuple = (1, 2, 4),  # Reduced depth
        unet_num_res_blocks: int = 1,  # Reduced from default 2
        unet_attention_resolutions: tuple = (),  # Disable attention to save memory
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.scaling_factor = scaling_factor

        # --- 1. Load the pre-trained VAE and freeze it ---
        self.vae = CfdVaeLite(in_chan=self.out_chan, out_chan=self.out_chan, latent_dim=latent_dim)
        self.vae.load_state_dict(torch.load(vae_weights_path, map_location="cpu"))
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # --- 2. Initialize the Diffusion U-Net with EXPLICIT memory-efficient parameters ---
        print(f"Initializing U-Net with optimized parameters:")
        print(f"  • Base channels: {unet_base_channels}")
        print(f"  • Channel multipliers: {unet_channel_mult}")
        print(f"  • Num res blocks: {unet_num_res_blocks}")
        print(f"  • Attention resolutions: {unet_attention_resolutions}")
        print(f"  • Latent size: {self.vae.latent_spatial_size}x{self.vae.latent_spatial_size}")
        
        self.unet = UNet2DConditionModel(
            sample_size=self.vae.latent_spatial_size,
            in_channels=latent_dim, 
            out_channels=latent_dim,
            # CRITICAL: Explicit architecture parameters for memory efficiency
            layers_per_block=unet_num_res_blocks,
            block_out_channels=tuple(unet_base_channels * m for m in unet_channel_mult),
            down_block_types=tuple(["DownBlock2D"] * len(unet_channel_mult)),
            up_block_types=tuple(["UpBlock2D"] * len(unet_channel_mult)),
            # Cross-attention configuration
            cross_attention_dim=self.in_chan + self.n_case_params,
            attention_head_dim=8,  # Smaller attention heads
            # Disable mid-block attention if not needed
            only_cross_attention=False,
            # Additional memory optimizations
            use_linear_projection=False,  # Slightly faster, less memory
            resnet_time_scale_shift="default",
        )

        # Enable gradient checkpointing to save VRAM
        self.unet.enable_gradient_checkpointing()
        
        # Print model size
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"  • U-Net parameters: {total_params / 1e6:.2f}M")

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )

    def forward(
        self, inputs: Tensor, case_params: Tensor, label: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Training forward pass.

        Args:
            inputs: Input velocity field [B, in_chan, H, W]
            case_params: Case parameters [B, n_case_params]
            label: Target velocity field [B, out_chan, H, W]
            mask: Optional mask tensor [H, W] or [B, 1, H, W]

        Returns:
            Dictionary containing predictions and losses
        """
        if label is None:
            raise ValueError("LDM requires a label for training.")

        # Validate input shapes
        assert inputs.shape[1] == self.in_chan, f"Expected {self.in_chan} input channels, got {inputs.shape[1]}"
        assert label.shape[1] == self.out_chan, f"Expected {self.out_chan} output channels, got {label.shape[1]}"
        assert case_params.shape[1] == self.n_case_params, f"Expected {self.n_case_params} case params, got {case_params.shape[1]}"

        batch_size = inputs.shape[0]
        device = inputs.device

        # Step 1: Encode the clean target image into the latent space
        # We only need the mean of the latent distribution for training
        with torch.no_grad():
            # Call the encoder explicitly to get the latent distribution
            target_latents_dist = self.vae.vae.encode(label).latent_dist
            target_latents = target_latents_dist.sample() * self.scaling_factor
            
        # Step 2: Add noise to the latents
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Prepare conditioning signal
        case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inputs.shape[2], inputs.shape[3])
        conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
        conditioning_signal = conditioning_signal.view(batch_size, self.in_chan + self.n_case_params, -1)
        # Permute the dimensions to [Batch, SequenceLength, FeatureDimension]
        conditioning_signal = conditioning_signal.permute(0, 2, 1)

        # Forward pass through U-Net
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
        """
        Generate next frame using diffusion sampling.

        Args:
            inputs: Input velocity field [B, in_chan, H, W]
            case_params: Case parameters [B, n_case_params]
            mask: Optional mask tensor
            num_inference_steps: Number of denoising steps

        Returns:
            Generated velocity field [B, out_chan, H, W]
        """
        batch_size = inputs.shape[0]

        # Prepare conditioning signal
        case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, inputs.shape[2], inputs.shape[3])
        conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
        conditioning_signal = conditioning_signal.view(batch_size, self.in_chan + self.n_case_params, -1)
        conditioning_signal = conditioning_signal.permute(0, 2, 1)

        # Start with random noise in the latent space
        latents = torch.randn(
            (batch_size, self.vae.vae.latent_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            device=inputs.device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop in latent space
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(
                sample=latents,
                timestep=t,
                encoder_hidden_states=conditioning_signal
            ).sample
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the clean latents back to a full-resolution image
        latents = latents / self.scaling_factor  # Unscale
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
    

class LatentDiffusionCfdModelLite(AutoCfdModel):
    """
    Lightweight Latent Diffusion Model using UNet2DModel.
    
    Key design choice: The U-Net ONLY sees noisy latents during training.
    Conditioning (velocity fields + case params) is encoded separately and 
    added as a learned bias to the latent space.
    
    This is more aligned with standard diffusion model theory:
    - x_t (noisy latents) is what we're denoising
    - c (conditioning) guides the denoising process
    - Model learns: p(x_0 | x_t, c)
    """
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
        scaling_factor: float = 4.5578,
        # Memory-efficient U-Net parameters
        unet_base_channels: int = 128,
        unet_channel_mult: tuple = (1, 2, 4, 4),
        unet_num_res_blocks: int = 2,
        unet_attention_resolutions: tuple = (4,),
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.latent_dim = latent_dim
        self.scaling_factor = scaling_factor

        # --- 1. Load the pre-trained VAE and freeze it ---
        self.vae = CfdVaeLite(in_chan=self.out_chan, out_chan=self.out_chan, latent_dim=latent_dim)
        self.vae.load_state_dict(torch.load(vae_weights_path, map_location="cpu"))
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        latent_size = self.vae.latent_spatial_size
        
        # --- 2. Conditioning encoder network ---
        # This encodes the conditioning signals (velocity + case params) into the latent space
        # Output has same shape as latent: [B, latent_dim, latent_size, latent_size]
        
        # First, project velocity inputs from 64x64 to latent_size x latent_size
        self.velocity_encoder = nn.Sequential(
            nn.Conv2d(in_chan, 64, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.SiLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),      # 32->16
            nn.SiLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),     # 16->8
            nn.SiLU(),
            nn.GroupNorm(8, 128),
        )
        
        # Project case parameters to spatial features
        self.case_param_mlp = nn.Sequential(
            nn.Linear(n_case_params, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        
        # Combine velocity features + case param features → latent space conditioning
        self.cond_combiner = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1),  # Output: latent_dim channels
        )

        # --- 3. Initialize the U-Net (operates only on latent space) ---
        print(f"Initializing UNet2DModel:")
        print(f"  • Latent size: {latent_size}x{latent_size}")
        print(f"  • Latent channels: {latent_dim}")
        print(f"  • Base channels: {unet_base_channels}")
        print(f"  • Channel multipliers: {unet_channel_mult}")
        
        self.unet = UNet2DModel(
            sample_size=latent_size,
            in_channels=latent_dim,   # Only the latent channels
            out_channels=latent_dim,  # Predict noise in latent space
            layers_per_block=unet_num_res_blocks,
            block_out_channels=tuple(unet_base_channels * m for m in unet_channel_mult),
            down_block_types=tuple(["DownBlock2D"] * len(unet_channel_mult)),
            up_block_types=tuple(["UpBlock2D"] * len(unet_channel_mult)),
            attention_head_dim=8 if unet_attention_resolutions else None,
        )

        # Enable gradient checkpointing
        if hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
        
        # Print model size
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        unet_params = sum(p.numel() for p in self.unet.parameters())
        cond_params = total_params - unet_params
        print(f"  • U-Net parameters: {unet_params / 1e6:.2f}M")
        print(f"  • Conditioning network: {cond_params / 1e6:.2f}M")
        print(f"  • Total trainable: {total_params / 1e6:.2f}M")

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )

    def encode_conditioning(self, inputs: Tensor, case_params: Tensor) -> Tensor:
        """
        Encode conditioning signals (velocity fields + case parameters) into latent space.
        
        Returns:
            conditioning: [B, latent_dim, latent_size, latent_size] - same shape as latents
        """
        batch_size = inputs.shape[0]
        latent_size = self.vae.latent_spatial_size
        
        # Encode velocity inputs to latent spatial dimensions
        velocity_features = self.velocity_encoder(inputs)  # [B, 128, 8, 8]
        
        # Process case parameters through MLP
        case_features = self.case_param_mlp(case_params)  # [B, 256]
        
        # Broadcast case features to spatial dimensions
        case_features_spatial = case_features.view(batch_size, 256, 1, 1).expand(
            -1, -1, latent_size, latent_size
        )  # [B, 256, 8, 8]
        
        # Concatenate and process to get conditioning in latent space
        combined = torch.cat([velocity_features, case_features_spatial], dim=1)  # [B, 384, 8, 8]
        conditioning = self.cond_combiner(combined)  # [B, latent_dim, 8, 8]
        
        return conditioning

    def forward(
        self, inputs: Tensor, case_params: Tensor, label: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Training forward pass.

        Args:
            inputs: Input velocity field [B, in_chan, H, W]
            case_params: Case parameters [B, n_case_params]
            label: Target velocity field [B, out_chan, H, W]
            mask: Optional mask tensor

        Returns:
            Dictionary containing predictions and losses
        """
        if label is None:
            raise ValueError("LDM requires a label for training.")

        # Validate input shapes
        assert inputs.shape[1] == self.in_chan, f"Expected {self.in_chan} input channels, got {inputs.shape[1]}"
        assert label.shape[1] == self.out_chan, f"Expected {self.out_chan} output channels, got {label.shape[1]}"
        assert case_params.shape[1] == self.n_case_params, f"Expected {self.n_case_params} case params, got {case_params.shape[1]}"

        batch_size = inputs.shape[0]
        device = inputs.device

        # Step 1: Encode the clean target image into the latent space
        with torch.no_grad():
            target_latents_dist = self.vae.vae.encode(label).latent_dist
            target_latents = target_latents_dist.sample() * self.scaling_factor
            
        # Step 2: Add noise to the latents
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Step 3: Encode conditioning into latent space
        conditioning = self.encode_conditioning(inputs, case_params)
        
        # Step 4: Add conditioning as a bias to the noisy latents
        # This is a form of "conditioning by addition" 
        conditioned_latents = noisy_latents + conditioning
        
        # Step 5: U-Net predicts the noise (not the clean signal)
        # The U-Net only sees the combined signal, learns to denoise it
        noise_pred = self.unet(conditioned_latents, timesteps).sample

        loss = F.mse_loss(noise_pred, noise)

        return {
            "preds": noise_pred,
            "loss": {"mse": loss, "nmse": loss / (torch.square(noise).mean() + 1e-8)}
        }

    @torch.no_grad()
    def generate(
        self, inputs: Tensor, case_params: Tensor, mask: Optional[Tensor] = None, num_inference_steps: int = 50
    ) -> Tensor:
        """
        Generate next frame using diffusion sampling.

        Args:
            inputs: Input velocity field [B, in_chan, H, W]
            case_params: Case parameters [B, n_case_params]
            mask: Optional mask tensor
            num_inference_steps: Number of denoising steps

        Returns:
            Generated velocity field [B, out_chan, H, W]
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Encode conditioning once (it's the same for all denoising steps)
        conditioning = self.encode_conditioning(inputs, case_params)

        # Start with random noise in the latent space
        latents = torch.randn(
            (batch_size, self.latent_dim, self.unet.config.sample_size, self.unet.config.sample_size),
            device=device
        )

        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Denoising loop in latent space
        for t in self.noise_scheduler.timesteps:
            # Add conditioning to current latents
            conditioned_latents = latents + conditioning

            # Predict noise
            noise_pred = self.unet(conditioned_latents, t).sample

            # Denoise one step
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the clean latents back to full-resolution image
        latents = latents / self.scaling_factor  # Unscale
        image = self.vae.vae.decode(latents).sample
        return image

    def generate_many(
        self, inputs: Tensor, case_params: Tensor, mask: Tensor, steps: int
    ) -> List[Tensor]:
        if inputs.dim() == 3:
            inputs, case_params, mask = inputs.unsqueeze(0), case_params.unsqueeze(0), mask.unsqueeze(0)
            
        generated_frames = []
        current_frame = inputs
        print(f"Generating {steps} frames with Latent Diffusion (Lite)...")
        for _ in tqdm(range(steps)):
            next_frame = self.generate(current_frame, case_params, mask)
            if mask is not None:
                next_frame = next_frame * mask
            generated_frames.append(next_frame)
            current_frame = next_frame
        return generated_frames