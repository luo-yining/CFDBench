from tap import Tap
from pathlib import Path
from typing import List

class Args(Tap):
    """
    Configuration for CFDBench training and evaluation.

    Parameters are organized into sections:
    1. General Settings
    2. Training Configuration
    3. Dataset Configuration
    4. Model Selection & Common Settings
    5. Model-Specific Hyperparameters
    6. Advanced Training Options
    """

    # ============================================================================
    # 1. GENERAL SETTINGS
    # ============================================================================

    mode: str = "train"
    """Mode: 'train' or 'test'"""

    seed: int = 0
    """Random seed for reproducibility"""

    output_dir: str = "result"
    """Directory to save results, checkpoints, and logs"""


    # ============================================================================
    # 2. TRAINING CONFIGURATION
    # ============================================================================

    # --- Optimization ---
    lr: float = 1e-4
    """Initial learning rate"""

    weight_decay: float = 1e-5
    """Weight decay for optimizer (L2 regularization)"""

    num_epochs: int = 100
    """Number of training epochs"""

    batch_size: int = 8
    """Training batch size"""

    eval_batch_size: int = 16
    """Evaluation batch size (can be different from training)"""

    # --- Learning Rate Scheduling ---
    lr_scheduler_factor: float = 0.5
    """Factor to reduce learning rate by on plateau"""

    lr_scheduler_patience: int = 5
    """Epochs to wait before reducing learning rate"""

    # --- Loss Function ---
    loss_name: str = "mse"
    """Loss function: 'mse', 'nmse', 'mae', or 'nmae'"""

    # --- Logging & Checkpointing ---
    log_interval: int = 50
    """Log training metrics every N batches"""

    eval_interval: int = 2
    """Evaluate model every N epochs"""

    save_checkpoint_every_n_epochs: int = 20
    """Save full checkpoint every N epochs (must be multiple of eval_interval). Best model is always saved."""

    save_images_every_n_epochs: int = 20
    """Save visualization images only every N epochs (must be multiple of eval_interval)"""

    # --- Early Stopping ---
    early_stopping_patience: int = 20
    """Epochs to wait for validation improvement before stopping"""

    early_stopping_delta: float = 1e-5
    """Minimum validation loss change to count as improvement"""


    # ============================================================================
    # 3. DATASET CONFIGURATION
    # ============================================================================

    data_name: str = "cylinder_geo"
    """
    Dataset name format: '<problem>_<subsets>'
    Problems: cavity, tube, dam, cylinder
    Subsets: bc (boundary conditions), geo (geometry), prop (properties)
    Examples: 'cylinder_geo', 'cavity_prop_bc_geo', 'dam_prop_geo'
    """

    data_dir: str = "../data"
    """Root directory containing CFDBench datasets"""

    num_rows: int = 64
    """Grid height (number of rows in spatial lattice)"""

    num_cols: int = 64
    """Grid width (number of columns in spatial lattice)"""

    delta_time: float = 0.1
    """Time step size for autoregressive models"""

    norm_props: int = 1
    """Whether to normalize physical properties (0=no, 1=yes)"""

    norm_bc: int = 1
    """Whether to normalize boundary conditions (0=no, 1=yes)"""


    # ============================================================================
    # 4. MODEL SELECTION & COMMON SETTINGS
    # ============================================================================

    model: str = "pixel_diffusion"
    """
    Model architecture to use:

    Autoregressive models (train_auto.py, train_auto_v2.py):
      - auto_ffn, auto_deeponet, auto_edeeponet, auto_deeponet_cnn
      - unet, fno, resnet
      - pixel_diffusion, latent_diffusion, latent_diffusion2

    Non-autoregressive models (train.py):
      - ffn, deeponet
    """

    in_chan: int = 2
    """Number of input channels (e.g., 2 for u,v velocity components)"""

    out_chan: int = 2
    """Number of output channels"""


    # ============================================================================
    # 5. MODEL-SPECIFIC HYPERPARAMETERS
    # ============================================================================

    # --- FFN (Feed-Forward Network) ---
    ffn_depth: int = 8
    """Number of hidden layers in FFN"""

    ffn_width: int = 100
    """Width of hidden layers in FFN"""

    # --- Auto-FFN (Autoregressive FFN) ---
    autoffn_depth: int = 8
    """Number of hidden layers in autoregressive FFN"""

    autoffn_width: int = 200
    """Width of hidden layers in autoregressive FFN"""

    # --- DeepONet ---
    deeponet_width: int = 100
    """Hidden layer width for DeepONet branch and trunk networks"""

    branch_depth: int = 8
    """Number of layers in DeepONet branch network"""

    trunk_depth: int = 8
    """Number of layers in DeepONet trunk network"""

    act_fn: str = "relu"
    """Activation function for DeepONet: 'relu', 'tanh', 'gelu', etc."""

    act_scale_invariant: int = 1
    """Use scale-invariant activation (0=no, 1=yes)"""

    act_on_output: int = 0
    """Apply activation to output layer (0=no, 1=yes)"""

    # --- Auto-EDeepONet (Enhanced DeepONet) ---
    autoedeeponet_width: int = 100
    """Hidden layer width for Auto-EDeepONet"""

    autoedeeponet_depth: int = 8
    """Number of layers in Auto-EDeepONet"""

    autoedeeponet_act_fn: str = "relu"
    """Activation function for Auto-EDeepONet"""

    # --- FNO (Fourier Neural Operator) ---
    fno_depth: int = 4
    """Number of Fourier layers in FNO"""

    fno_hidden_dim: int = 32
    """Hidden dimension for FNO"""

    fno_modes_x: int = 12
    """Number of Fourier modes in x-direction"""

    fno_modes_y: int = 12
    """Number of Fourier modes in y-direction"""

    # --- U-Net ---
    unet_dim: int = 12
    """Base channel dimension for U-Net"""

    unet_insert_case_params_at: str = "input"
    """Where to inject case parameters: 'input' or 'bottleneck'"""

    # --- ResNet ---
    resnet_depth: int = 4
    """Number of residual blocks in ResNet"""

    resnet_hidden_chan: int = 16
    """Number of hidden channels in ResNet"""

    resnet_kernel_size: int = 7
    """Convolution kernel size for ResNet"""

    resnet_padding: int = 3
    """Padding for ResNet convolutions"""

    # --- VAE (Variational Autoencoder) ---
    vae_kl_weight: float = 1e-4
    """Weight for KL divergence loss in VAE"""

    vae_kl_annealing_epochs: int = 20
    """Epochs to linearly anneal KL weight from 0 to final value"""

    # VAE Architecture (AutoencoderKL ddconfig)
    double_z: bool = True
    """Use double z channels in VAE"""

    z_channels: int = 4
    """Number of latent channels in VAE"""

    resolution: int = 64
    """Input resolution for VAE"""

    in_channels: int = 2
    """Input channels for VAE (use in_chan instead when possible)"""

    out_ch: int = 2
    """Output channels for VAE (use out_chan instead when possible)"""

    ch: int = 64
    """Base channel count for VAE"""

    ch_mult: List[int] = [1, 2, 3, 4]
    """Channel multipliers for VAE encoder/decoder stages"""

    num_res_blocks: int = 2
    """Number of residual blocks per VAE stage"""

    attn_resolutions: List[int] = [16, 8]
    """Resolutions at which to apply attention in VAE"""

    dropout: float = 0.0
    """Dropout rate for VAE"""

    has_mid_attn: bool = True
    """Use attention in VAE middle block"""

    embed_dim: int = 4
    """Embedding dimension for VAE latent space"""

    # VAE Loss Configuration
    disc_start: int = 50001
    """Training step to start discriminator"""

    kl_weight: float = 0.000001
    """KL divergence weight for VAE loss"""

    disc_weight: float = 0.5
    """Discriminator loss weight"""

    # --- Latent Diffusion Model (LDM) ---
    project_root = Path(__file__).parent.parent
    default_vae_path = project_root / "weights" / "vaelite_002.pt"

    ldm_vae_weights_path: str = str(default_vae_path)
    """Path to pre-trained VAE weights for latent diffusion"""

    ldm_latent_dim: int = 4
    """Number of channels in latent space"""

    ldm_noise_scheduler_timesteps: int = 1000
    """Number of diffusion timesteps"""

    ldm_scaling_factor: float = 4.5578
    """Scaling factor for VAE latent space (pre-calculated)"""

    # U-Net for LDM2
    unet_base_channels: int = 64
    """Base channel count for LDM2 U-Net"""

    unet_channel_mult = (1, 2, 4)
    """Channel multipliers for LDM2 U-Net stages"""

    unet_num_res_blocks: int = 1
    """Number of residual blocks per LDM2 U-Net stage"""

    unet_attention_resolutions = ()
    """Resolutions for attention in LDM2 U-Net"""

    # --- Pixel Diffusion (PUNetG) ---
    pixel_diffusion_base_channels: int = 64
    """Base channel count for Pixel Diffusion PUNetG model"""

    pixel_diffusion_channel_mults: tuple = (1, 2, 4)
    """Channel multipliers for PUNetG stages"""

    pixel_diffusion_num_res_blocks: int = 2
    """Number of residual blocks per PUNetG stage"""

    pixel_diffusion_dropout: float = 0.1
    """Dropout rate for PUNetG ResNet blocks"""


    # ============================================================================
    # 6. ADVANCED TRAINING OPTIONS
    # ============================================================================

    use_mixed_precision: bool = True
    """Enable Automatic Mixed Precision (AMP) for faster training and lower memory"""

    gradient_accumulation_steps: int = 1
    """
    Accumulate gradients over N steps before updating weights.
    Effective batch size = batch_size × gradient_accumulation_steps
    """

    use_gradient_checkpointing: bool = False
    """
    Enable gradient checkpointing for diffusion models.
    Trades compute for memory (slower but uses less VRAM)
    """

    clear_cache: bool = False
    """Clear CUDA cache between batches (slower but can prevent OOM)"""


    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def get_ddconfig(self):
        """Creates a ddconfig object for AutoencoderKL from arguments."""
        from diffsci.models.nets.autoencoderldm2d import ddconfig

        return ddconfig(
            double_z=self.double_z,
            z_channels=self.z_channels,
            resolution=self.resolution,
            in_channels=self.in_channels,
            out_ch=self.out_ch,
            ch=self.ch,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            has_mid_attn=self.has_mid_attn
        )

    def get_lossconfig(self):
        """Creates a lossconfig object for AutoencoderKL from arguments."""
        from diffsci.models.nets.autoencoderldm2d import lossconfig

        return lossconfig(
            disc_start=self.disc_start,
            kl_weight=self.kl_weight,
            disc_weight=self.disc_weight,
        )


def is_args_valid(args: Args):
    """Validate argument values."""
    assert any(
        key in args.data_name for key in ["poiseuille", "cavity", "karman", "tube", "dam", "cylinder"]
    )
    assert args.batch_size > 0
    # Model validation removed - let init_model() handle this
