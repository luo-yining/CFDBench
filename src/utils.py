import json
from pathlib import Path
from typing import Union, Optional

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

from args import Args


def plot_contour(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    plt.tricontourf(x, y, z)
    plt.colorbar()
    plt.show()


def dump_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load a JSON object from a file"""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def plot_predictions(
    label: Tensor,
    pred: Tensor,
    out_dir: Path,
    step: int,
    inp: Optional[Tensor] = None,  # non-autoregressive input func. is not plottable.
):
    assert all([isinstance(x, Tensor) for x in [label, pred]])
    assert (
        label.shape == pred.shape
    ), f"{label.shape}, {pred.shape}"

    if inp is not None:
        assert inp.shape == label.shape
        assert isinstance(inp, Tensor)
        inp_dir = out_dir / "input"
        inp_dir.mkdir(exist_ok=True, parents=True)
        inp_arr = inp.cpu().detach().numpy()
    label_dir = out_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True, parents=True)

    pred_arr = pred.cpu().detach().numpy()
    label_arr = label.cpu().detach().numpy()

    # Plot and save images
    if inp is not None:
        u_min = min(inp_arr.min(), pred_arr.min(), label_arr.min())
        u_max = max(inp_arr.max(), pred_arr.max(), label_arr.max())
    else:
        u_min = min(pred_arr.min(), label_arr.min())
        u_max = max(pred_arr.max(), label_arr.max())

    if inp is not None:
        plt.axis('off')
        plt.imshow(inp_arr, vmin=inp_arr.min(), vmax=inp_arr.max(), cmap="coolwarm")
        plt.savefig(inp_dir / f"{step:04}.png", bbox_inches='tight', pad_inches=0)
        plt.clf()

    plt.axis('off')
    plt.imshow(label_arr, vmin=label_arr.min(), vmax=label_arr.max(), cmap="coolwarm")
    plt.savefig(label_dir / f"{step:04}.png", bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.axis('off')
    plt.imshow(pred_arr, vmin=pred_arr.min(), vmax=pred_arr.max(), cmap="coolwarm")
    plt.savefig(pred_dir / f"{step:04}.png", bbox_inches='tight', pad_inches=0)
    plt.clf()


def plot(inp: Tensor, label: Tensor, pred: Tensor, out_path: Path):
    assert all([isinstance(x, Tensor) for x in [inp, label, pred]])
    assert (
        inp.shape == label.shape == pred.shape
    ), f"{inp.shape}, {label.shape}, {pred.shape}"

    tensor_dir = out_path.parent / "tensors"
    tensor_dir.mkdir(exist_ok=True, parents=True)
    tensor_path = tensor_dir / (out_path.stem + ".pt")
    torch.save((inp, label, pred), tensor_path)

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    plt.subplots_adjust(
        left=0.0, right=1, bottom=0.0, top=1, wspace=0, hspace=0
    )
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]

    axs = axs.flatten()

    last_im = None

    inp_arr = inp.cpu().detach().numpy()
    pred_arr = pred.cpu().detach().numpy()
    label_arr = label.cpu().detach().numpy()
    u_min = min(inp_arr.min(), pred_arr.min(), label_arr.min())
    u_max = max(inp_arr.max(), pred_arr.max(), label_arr.max())

    def sub_plot(idx, data: np.ndarray, title):
        nonlocal last_im
        ax = axs[idx - 1]
        ax.set_axis_off()
        im = ax.imshow(data, vmin=u_min, vmax=u_max, cmap="coolwarm")
        # fig.colorbar(im, ax=ax)
        # ax.set_title(title)
        last_im = im

    # print("plotting input")
    sub_plot(1, inp_arr, "Input")
    sub_plot(2, label_arr, "Label")
    sub_plot(3, pred_arr, "Prediction")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    fig.colorbar(last_im, cax=cbar_ax)
    # # Add a common colorbar
    # fig.colorbar(last_im, cax=cbar_ax)

    # Add some spacing between the subplots
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_loss(losses, out: Path, fontsize: int = 12, linewidth: int = 2):
    plt.plot(losses, linewidth=linewidth)
    plt.xlabel("Step", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.savefig(out)
    plt.clf()
    plt.close()


def get_best_ckpt(output_dir: Path) -> Union[Path, None]:
    """
    Returns None if there is no ckpt-* directory in output_dir
    """
    ckpt_dirs = sorted(output_dir.glob("ckpt-*"))
    best_loss = float("inf")
    best_ckpt_dir = None
    for ckpt_dir in ckpt_dirs:
        scores = load_json(ckpt_dir / "scores.json")
        dev_loss = scores["dev_loss"]
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_ckpt_dir = ckpt_dir
    return best_ckpt_dir


def load_ckpt(model, ckpt_path: Path) -> None:
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))


def get_output_dir(args: Args, is_auto: bool = False) -> Path:
    print(args.output_dir)
    output_dir = Path(
        args.output_dir,
        "auto" if is_auto else "non-auto",
        args.data_name,
        f"dt{args.delta_time}",
        args.model,
    )
    if args.model == "deeponet":
        dir_name = (
            f"lr{args.lr}"
            + f"_width{args.deeponet_width}"
            + f"_depthb{args.branch_depth}"
            + f"_deptht{args.trunk_depth}"
            + f"_normprop{args.norm_props}"
            + f"_act{args.act_fn}"
            + f"-{args.act_scale_invariant}"
            + f"-{args.act_on_output}"
        )
        output_dir /= dir_name
        return output_dir
    elif args.model == "unet":
        dir_name = (
            f"lr{args.lr}" f"_d{args.unet_dim}" f"_cp{args.unet_insert_case_params_at}"
        )
        output_dir /= dir_name
        return output_dir
    elif args.model == "fno":
        dir_name = (
            f"lr{args.lr}"
            + f"_d{args.fno_depth}"
            + f"_h{args.fno_hidden_dim}"
            + f"_m1{args.fno_modes_x}"
            + f"_m2{args.fno_modes_y}"
        )
        output_dir /= dir_name
        return output_dir
    elif args.model == "resnet":
        dir_name = (
            f"lr{args.lr}" f"_d{args.resnet_depth}" f"_w{args.resnet_hidden_chan}"
        )
        return output_dir / dir_name
    elif args.model == "auto_edeeponet":
        dir_name = (
            f"lr{args.lr}"
            + f"_width{args.autoedeeponet_width}"
            + f"_depthb{args.autoedeeponet_depth}"
            + f"_deptht{args.autoedeeponet_depth}"
            + f"_normprop{args.norm_props}"
            + f"_act{args.autoedeeponet_act_fn}"
            # + f"-{args.act_scale_invariant}"
            # + f"-{args.act_on_output}"
        )
        return output_dir / dir_name
    elif args.model == "auto_deeponet":
        dir_name = (
            f"lr{args.lr}"
            f"_width{args.deeponet_width}"
            f"_depthb{args.branch_depth}"
            f"_deptht{args.trunk_depth}"
            f"_normprop{args.norm_props}"
            f"_act{args.act_fn}"
        )
        return output_dir / dir_name
    elif args.model == "auto_ffn":
        dir_name = (
            f"lr{args.lr}" f"_width{args.autoffn_width}" f"_depth{args.autoffn_depth}"
        )
        return output_dir / dir_name
    elif args.model == "auto_deeponet_cnn":
        dir_name = f"lr{args.lr}" f"_depth{args.autoffn_depth}"
        return output_dir / dir_name
    elif args.model == "ffn":
        dir_name = f"lr{args.lr}" f"_width{args.ffn_width}" f"_depth{args.ffn_depth}"
        return output_dir / dir_name
    else:
        raise NotImplementedError


def load_best_ckpt(model, output_dir: Path):
    print(f"Finding the best checkpoint from {output_dir}")
    best_ckpt_dir = get_best_ckpt(output_dir)
    assert best_ckpt_dir is not None
    print(f"Loading best checkpoint from {best_ckpt_dir}")
    ckpt_path = best_ckpt_dir / "model.pt"
    load_ckpt(model, ckpt_path)
