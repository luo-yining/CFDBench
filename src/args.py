from tap import Tap
from pathlib import Path


class Args(Tap):
    seed: int = 0

    output_dir: str = "result"
    """The directory to save the results to"""

    lr: float = 1e-4  # Initial learning rate

    lr_step_size: int = 20  # LR decays every lr_step_size epochs

    num_epochs: int = 100  # Number of epochs to train for

    eval_interval: int = 10
    """Evaluate every eval_interval epochs, and save checkpoint."""

    log_interval: int = 50  # Log training progress every log_interval batches

    loss_name: str = "mse"
    """
    The loss function to use for training.
    Choices: ['mse', 'nmse', 'mae', 'nmae'].
    """

    mode: str = "train"
    """"train" or "test" for train/test only"""

    model: str = "deeponet"
    """
    For autoregressive modeling (`train_auto.py`), it must be one of: ['auto_ffn', 'auto_deeponet', 'auto_edeeponet', 'auto_deeponet_cnn', 'unet', 'fno', 'resnet'],
    for non-autoregressive modeling (`train.py`), it must be one of: ['ffn', 'deeponet'].
    """
    in_chan: int = 2
    """Number of input channels, only applicable to autoregressive models"""
    out_chan: int = 2
    """Number of output channels, only applicable to autoregressive models"""

    batch_size: int = 16
    eval_batch_size: int = 16

    # ------------  Dataset hyperparamters ----------------

    data_name: str = "cylinder_bc_geo"
    """
    One of: 'laminar_*', 'cavity_*', 'karman_*', where * is used to
    indicate the subset to use. E.g., 'laminar_prop_geo' trains
    on the subset of laminar task with varying geometry and physical
    properties.
    """

    data_dir: str = "../data"
    """The directory that contains the CFDBench."""

    norm_props: int = 1
    """Whether to normalize the physical properties."""
    
    norm_bc: int = 1
    """Whether to normalize the boundary conditions."""

    num_rows = 64
    """Number of rows in the lattice that represents the field."""

    num_cols = 64
    """Number of columns in the lattice that represents the field."""

    delta_time: float = 0.1
    """The time step size."""

    # FFN hyperparameters
    ffn_depth: int = 8
    ffn_width: int = 100

    # DeepONet hyperparameters
    deeponet_width: int = 100
    branch_depth: int = 8
    trunk_depth: int = 8
    act_fn: str = "relu"
    act_scale_invariant: int = 1
    act_on_output: int = 0

    # Auto-FFN hyperparameters
    autoffn_depth: int = 8
    autoffn_width: int = 200

    # Auto-EDeepONet hyperparameters
    autoedeeponet_width: int = 100
    autoedeeponet_depth: int = 8
    autoedeeponet_act_fn: str = "relu"
    # autoedeeponet_act_scale_invariant: int = 1
    # autoedeeponet_act_on_output: int = 0

    # FNO hyperparameters
    fno_depth: int = 4
    fno_hidden_dim: int = 32
    fno_modes_x: int = 12
    fno_modes_y: int = 12

    # UNet
    unet_dim: int = 12
    unet_insert_case_params_at: str = "input"

    # ResNet hyperparameters
    resnet_depth: int = 4
    resnet_hidden_chan: int = 16
    resnet_kernel_size: int = 7
    resnet_padding: int = 3

    # Early Stopping Parameters
    early_stopping_patience: int = 10
    """Number of epochs to wait for validation loss to improve before stopping."""
    early_stopping_delta: float = 0.0001
    """Minimum change in validation loss to be considered an improvement."""

    
    # --- Latent Diffusion Model and VAE Hyperparameters ---
    
    # clear cache argument
    clear_cache = False
    # Programmatically find the project's root directory.
    # Path(__file__) is the path to args.py
    # .parent is src/, .parent.parent is the CFDBench/ root.
    project_root = Path(__file__).parent.parent 
    
    # Construct the default path to the VAE weights
    default_vae_path = project_root / "weights" / "vae3_cylinder_bc_geo.pt"

    # Define the command-line argument, converting the Path object to a string
    ldm_vae_weights_path: str = str(default_vae_path)
    """Path to the pre-trained VAE weights."""
    
    ldm_latent_dim: int = 4
    """The number of channels in the latent space."""
    
    ldm_noise_scheduler_timesteps: int = 1000
    """The number of timesteps for the diffusion process."""
    
    vae_kl_weight: float = 1e-6
    """The weight of the KL Divergence loss for VAE training."""
    
    vae_kl_annealing_epochs: int = 20
    """Number of epochs to linearly ramp up the KL weight from 0 to its final value."""

def is_args_valid(args: Args):
    assert any(
        key in args.data_name for key in ["poiseuille", "cavity", "karman"]
    )
    assert args.batch_size > 0
    assert args.model in ["deeponet", "unet", "fno", "resnet"]
