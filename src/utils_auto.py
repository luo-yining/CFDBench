from models.resnet import ResNet
from models.unet import UNet
from models.base_model import AutoCfdModel
from models.auto_deeponet import AutoDeepONet
from models.auto_edeeponet import AutoEDeepONet
from models.auto_deeponet_cnn import AutoDeepONetCnn
from models.fno.fno2d import Fno2d
from models.auto_ffn import AutoFfn
from models.loss import loss_name_to_fn
from args import Args


def init_model(args: Args) -> AutoCfdModel:
    loss_fn = loss_name_to_fn(args.loss_name)
    if args.model == "resnet":
        model = ResNet(
            in_chan=args.in_chan,  # mask is not included
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            hidden_chan=args.resnet_hidden_chan,
            num_blocks=args.resnet_depth,
            kernel_size=args.resnet_kernel_size,
            padding=args.resnet_padding,
        ).cuda()
        return model
    elif args.model == "unet":
        model = UNet(
            in_chan=args.in_chan,  # Mask is not included
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            insert_case_params_at=args.unet_insert_case_params_at,
            dim=args.unet_dim,
        ).cuda()
        return model
    elif args.model == "auto_deeponet":
        if any(x in args.data_name for x in ['laminar', 'karman', "dam"]):
            num_rows = args.num_rows + 2  # Top and bottom boundaries
            num_cols = args.num_cols + 1  # Left boundary
        else:
            num_rows = args.num_rows
            num_cols = args.num_cols
        if 'karman' in args.data_name:
            # +8 for the case condition parameters
            # vel_in, density, viscosity, height, width, radius, center_x, center_y
            branch_dim = num_rows * num_cols + 8
        else:
            # +5 for the case condition parameters
            # vel_in, density, viscosity, height, width
            branch_dim = num_rows * num_cols + 5
        model = AutoDeepONet(
            branch_dim=branch_dim,  # +2 因为物性
            trunk_dim=2,  # (x, y)
            loss_fn=loss_fn,
            width=args.deeponet_width,
            trunk_depth=args.trunk_depth,
            branch_depth=args.branch_depth,
            act_name=args.act_fn,
        ).cuda()
        return model
    elif args.model == "fno":
        model = Fno2d(
            in_chan=args.in_chan,  # 2 for u and v
            out_chan=args.out_chan,
            loss_fn=loss_fn,
            num_layers=args.fno_depth,
            hidden_dim=args.fno_hidden_dim,  # Hidden dim. in the temporal domain
            modes1=args.fno_modes_x,
            modes2=args.fno_modes_y,
        ).cuda()
        return model
    elif args.model == "auto_edeeponet":
        if any(x in args.data_name for x in ['laminar', 'karman', "dam"]):
            num_rows = args.num_rows + 2  # Top and bottom boundaries
            num_cols = args.num_cols + 1  # Left boundary
        else:
            num_rows = args.num_rows
            num_cols = args.num_cols
        if 'karman' in args.data_name:
            # vel_in, density, viscosity, height, width, radius, center_x, center_y
            dim_branch2 = 8
        else:
            # vel_in, density, viscosity, height, width
            dim_branch2 = 5  # physical properties
        dim_branch1 = num_rows * num_cols  # u(t-1)
        model = AutoEDeepONet(
            dim_branch1=dim_branch1,
            dim_branch2=dim_branch2,
            trunk_dim=2,  # (x, y)
            loss_fn=loss_fn,
            width=args.autoedeeponet_width,
            trunk_depth=args.autoedeeponet_depth,
            branch_depth=args.autoedeeponet_depth,
            act_name=args.autoedeeponet_act_fn,
        ).cuda()
        return model
    elif args.model == "auto_ffn":
        if any(x in args.data_name for x in ['laminar', 'karman', "dam"]):
            num_rows = args.num_rows + 2  # Top and bottom boundaries
            num_cols = args.num_cols + 1  # Left boundary
        else:
            num_rows = args.num_rows
            num_cols = args.num_cols
        if 'karman' in args.data_name:
            # vel_in, density, viscosity, height, width, radius, center_x, center_y
            n_case_params = 8
        else:
            # vel_in, density, viscosity, height, width
            n_case_params = 5  # physical properties
        model = AutoFfn(
            input_field_dim=num_rows * num_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
            width=args.autoffn_width,
            depth=args.autoffn_depth,
        ).cuda()
        return model
    elif args.model == "auto_deeponet_cnn":
        if any(x in args.data_name for x in ['laminar', 'karman', "dam"]):
            num_rows = args.num_rows + 2  # Top and bottom boundaries
            num_cols = args.num_cols + 1  # Left boundary
        else:
            num_rows = args.num_rows
            num_cols = args.num_cols
        if 'karman' in args.data_name:
            # vel_in, density, viscosity, height, width, radius, center_x, center_y
            n_case_params = 8
        else:
            # vel_in, density, viscosity, height, width
            n_case_params = 5  # physical properties
        model = AutoDeepONetCnn(
            in_chan=3,  # (u, v, p)
            height=num_rows,
            width=num_cols,
            num_case_params=n_case_params,
            query_dim=2,
            loss_fn=loss_fn,
        ).cuda()
        return model
    else:
        raise ValueError(f"Invalid model name: {args.model}")
