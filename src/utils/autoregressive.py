from typing import Tuple
import torch.nn.functional as F

from models.resnet import ResNet
from models.unet import UNet
from models.base_model import AutoCfdModel
from models.auto_deeponet import AutoDeepONet
from models.auto_edeeponet import AutoEDeepONet
from models.auto_deeponet_cnn import AutoDeepONetCnn
from models.fno.fno2d import Fno2d
from models.auto_ffn import AutoFfn
from models.latent_diffusion import LatentDiffusionCfdModel
from models.ldm2 import LatentDiffusionCfdModel2
from models.pixel_diffusion import PixelDiffusionCfdModel
from models.loss import loss_name_to_fn
from args import Args


def get_input_shapes(args: Args) -> Tuple[int, int, int]:
    """
    Returns the number of rows, columns, and case parameters depending on the
    `data_name`, `num_rows` and `num_cols` attributes of `args`.
    """
    if any(x in args.data_name for x in ["tube", "dam", "cylinder"]):
        n_rows = args.num_rows + 2  # Top and bottom boundaries
        n_cols = args.num_cols + 1  # Left boundary
    else:
        assert "cavity" in args.data_name
        n_rows = args.num_rows
        n_cols = args.num_cols
    if "cylinder" in args.data_name:
        # vel_in, density, viscosity, height, width, radius, center_x, center_y
        n_case_params = 8
    else:
        assert any(x in args.data_name for x in ["cavity", "tube", "dam"])
        # vel_in, density, viscosity, height, width
        n_case_params = 5  # physical properties
    return n_rows, n_cols, n_case_params


def init_model(args: Args) -> AutoCfdModel:
    """
    All instances of autoregressive models go through this.
    """
    loss_fn = loss_name_to_fn(args.loss_name)
    n_rows, n_cols, n_case_params = get_input_shapes(args)

    if args.model == "auto_ffn":
        model = AutoFfn(
            input_field_dim=n_rows * n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
            width=args.autoffn_width,
            depth=args.autoffn_depth,
        )
        return model
    elif args.model == "auto_deeponet":
        branch_dim = n_cols * n_rows + n_case_params
        model = AutoDeepONet(
            branch_dim=branch_dim, 
            trunk_dim=2,  # (x, y)
            loss_fn=loss_fn,
            width=args.deeponet_width,
            trunk_depth=args.trunk_depth,
            branch_depth=args.branch_depth,
            act_name=args.act_fn,
        )
        return model
    elif args.model == "auto_edeeponet":
        model = AutoEDeepONet(
            dim_branch1=n_rows * n_cols,
            dim_branch2=n_case_params,
            trunk_dim=2,  # (x, y)
            loss_fn=loss_fn,
            width=args.autoedeeponet_width,
            trunk_depth=args.autoedeeponet_depth,
            branch_depth=args.autoedeeponet_depth,
            act_name=args.autoedeeponet_act_fn,
        ).cuda()
        return model
    elif args.model == "auto_deeponet_cnn":
        model = AutoDeepONetCnn(
            in_chan=args.in_chan,
            height=n_rows,
            width=n_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
        )
        return model
    elif args.model == "resnet":
        model = ResNet(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            hidden_chan=args.resnet_hidden_chan,
            num_blocks=args.resnet_depth,
            kernel_size=args.resnet_kernel_size,
            padding=args.resnet_padding,
        )
        return model
    elif args.model == "unet":
        model = UNet(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            insert_case_params_at=args.unet_insert_case_params_at,
            dim=args.unet_dim,
        )
        return model
    elif args.model == "fno":
        model = Fno2d(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            n_case_params=n_case_params,
            loss_fn=loss_fn,
            num_layers=args.fno_depth,
            hidden_dim=args.fno_hidden_dim,  # Hid. dim. in the temporal domain
            modes1=args.fno_modes_x,
            modes2=args.fno_modes_y,
        )
        return model
    elif args.model == "latent_diffusion":
        model = LatentDiffusionCfdModel(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            n_case_params=n_case_params,
            vae_weights_path=args.ldm_vae_weights_path,
            noise_scheduler_timesteps=args.ldm_noise_scheduler_timesteps,
            loss_fn=loss_fn,
            latent_dim=args.ldm_latent_dim,
            scaling_factor=args.ldm_scaling_factor,
        )
        return model
    elif args.model == 'latent_diffusion2':

        # Get U-Net architecture parameters with defaults
        unet_base_channels = getattr(args, 'unet_base_channels', 64)
        unet_channel_mult = getattr(args, 'unet_channel_mult', (1, 2, 4))
        unet_num_res_blocks = getattr(args, 'unet_num_res_blocks', 1)
        unet_attention_resolutions = getattr(args, 'unet_attention_resolutions', ())

        model = LatentDiffusionCfdModel2(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            vae_weights_path=args.ldm_vae_weights_path,
            image_size=64,
            latent_dim=args.ldm_latent_dim,
            noise_scheduler_timesteps=args.ldm_noise_scheduler_timesteps,
            scaling_factor=args.ldm_scaling_factor,
            # Pass the memory-efficient parameters
            unet_base_channels=unet_base_channels,
            unet_channel_mult=unet_channel_mult,
            unet_num_res_blocks=unet_num_res_blocks,
            unet_attention_resolutions=unet_attention_resolutions,
        )
        return model
    elif args.model == "pixel_diffusion":
        model = PixelDiffusionCfdModel(
            in_chan=args.in_chan,
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            n_case_params=n_case_params,
            image_size=64,
            noise_scheduler_timesteps=args.ldm_noise_scheduler_timesteps,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            base_channels=args.pixel_diffusion_base_channels,
            channel_mults=args.pixel_diffusion_channel_mults,
            num_res_blocks=args.pixel_diffusion_num_res_blocks,
            dropout=args.pixel_diffusion_dropout,
        )
        return model
    else:
        raise ValueError(f"Invalid model name: {args.model}")
