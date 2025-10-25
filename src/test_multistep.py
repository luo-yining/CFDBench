from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor, FloatTensor
from torch.nn import functional as F
import matplotlib.pyplot as plt

from args import Args
from utils.common import load_best_ckpt, get_output_dir, dump_json
from dataset import get_auto_dataset
from models.base_model import AutoCfdModel, CfdModel
from utils.autoregressive import init_model as init_auto_model
from models.ffn import FfnModel
from models.deeponet import DeepONet
from models.loss import loss_name_to_fn


def init_model(args: Args) -> CfdModel:
    print(f"Initting {args.model}")
    loss_fn = loss_name_to_fn(args.loss_name)
    query_coord_dim = 3  # (t, x, y)
    if "cylinder" in args.data_name:
        # (density, viscosity, u_top, h, w, radius, center_x, center_y)
        n_case_params = 8
    else:
        n_case_params = 5  # (density, viscosity, u_top, h, w)
    if args.model == "deeponet":
        model = DeepONet(
            branch_dim=n_case_params,
            trunk_dim=query_coord_dim,
            loss_fn=loss_fn,
            width=args.deeponet_width,
            trunk_depth=args.trunk_depth,
            branch_depth=args.branch_depth,
            act_name=args.act_fn,
            act_norm=bool(args.act_scale_invariant),
            act_on_output=bool(args.act_on_output),
        ).cuda()
    elif args.model == "ffn":
        widths = (
            [n_case_params + query_coord_dim]
            + [args.ffn_width] * args.ffn_depth
            + [1]
        )
        model = FfnModel(
            widths=widths,
            loss_fn=loss_fn,
        ).cuda()
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")
    return model


def plot_metrics(metrics: List[dict], out_path: Optional[Path] = None):
    metrics = np.array(metrics)  # type: ignore
    for key in ["nmse", "mse", "mae"]:
        values = [x[key] for x in metrics]
        plt.plot(values, label=key.upper())
    plt.legend()
    plt.xlabel("Steps")
    plt.yscale("log")
    # plt.title("Temporal extrapolation of Auto-DeepONet")
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches="tight")


def get_metrics(preds: Tensor, labels: Tensor):
    assert preds.shape == labels.shape, f"{preds.shape}, {labels.shape}"
    mse = ((preds - labels) ** 2).mean().detach().cpu().item()
    nmse = mse / ((labels**2).mean()).detach().cpu().item()
    mae = F.l1_loss(preds, labels).detach().cpu().item()
    return dict(
        mse=mse,
        nmse=nmse,
        mae=mae,
    )


def case_params_to_tensor(case_params_dict: dict):
    # Case params is a dict, turn it into a tensor
    keys = [
        x for x in case_params_dict.keys() if x not in ["rotated", "dx", "dy"]
    ]
    case_params_vec = [case_params_dict[k] for k in keys]
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    return case_params


def combine_dicts(dicts: List[dict]) -> dict:
    result = {}
    for key in dicts[0]:
        result[key] = np.mean([d[key] for d in dicts])
    return result


def infer_case(
    model: Union[AutoCfdModel, CfdModel],
    case_features: Tensor,
    case_params: Tensor,
    infer_steps: int,
):
    with torch.no_grad():
        if isinstance(model, AutoCfdModel):
            start_frame = case_features[0, :-1]
            mask = case_features[0, -1]
            preds = model.generate_many(
                inputs=start_frame,
                case_params=case_params,
                mask=mask,
                steps=infer_steps,
            )
            return preds
        elif isinstance(model, CfdModel):
            preds = []
            for step in range(infer_steps):
                t = torch.tensor([step], dtype=torch.float32).to("cuda")
                pred = model.generate_one(
                    case_params=case_params,
                    t=t,
                    height=case_features.shape[2],
                    width=case_features.shape[3],
                )
                preds.append(pred)
            return preds
        else:
            raise NotImplementedError


def infer(
    model,
    all_features: List[Tensor],
    all_case_params: List[Tensor],
    infer_steps: int,
):
    n_cases = len(all_features)
    print(f"Number of cases: {n_cases}")
    all_preds = []
    for case_id in range(n_cases):
        case_features = all_features[case_id]
        case_params = all_case_params[case_id]
        # (steps, c, h, w)
        case_pred = infer_case(model, case_features, case_params, infer_steps)
        all_preds.append(case_pred)

    # Compute metrics
    all_metrics = []
    for step in range(infer_steps):
        # print(step)
        step_metrics = []
        for case_id in range(n_cases):
            # The last channel is mask
            case_features = all_features[case_id][step]  # (c + 1, h, w)
            case_pred = all_preds[case_id][step]  # (b, c, h, w)
            # Assume `all_preds` has a batch size of 1

            preds = case_pred[0]  # (c, h, w)
            label = case_features[:-1]  # (c, h, w)
            mask = case_features[-1]  # (h, w)

            # Only compute the metrics for the u component
            preds = preds[0]
            label = label[0]

            preds = preds * mask  # (c, h, w)
            label = label * mask  # (c, h, w)
            metrics = get_metrics(preds, label)
            step_metrics.append(metrics)
        metrics = combine_dicts(step_metrics)
        print(metrics)
        all_metrics.append(metrics)
    return all_metrics


def main():
    args = Args().parse_args()
    print(args)

    # Load data
    data_dir = Path(args.data_dir)
    _, _, test_data = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        load_splits=["test"],
    )
    assert test_data is not None
    assert test_data.all_features is not None
    assert test_data.case_params is not None
    print("Test data size:", len(test_data))
    infer_steps = 20
    all_features = test_data.all_features
    all_case_params = test_data.case_params

    # Make sure each case has at least `infer_steps` steps by repeating
    # the last frame. This is because we assume that flow has
    # reached steady state.
    for case_id, case_features in enumerate(all_features):
        num_frames = case_features.shape[0]
        while num_frames < infer_steps:
            case_features = np.concatenate(
                [case_features, case_features[-1:]], axis=0
            )
            num_frames += 1
        all_features[case_id] = FloatTensor(case_features).to("cuda")

    # Turn case params into tensors
    for case_id, case_params in enumerate(all_case_params):
        all_case_params[case_id] = case_params_to_tensor(case_params).to(
            "cuda"
        )

    # print("Number of cases:", len(all_features))
    # exit()

    # Load model
    is_autoregressive = args.model not in ["deeponet", "ffn"]
    if args.model in ["deeponet", "ffn"]:
        model = init_model(args)
    else:
        model = init_auto_model(args)
    output_dir = get_output_dir(args, is_auto=is_autoregressive)
    load_best_ckpt(model, output_dir)

    print("====== Start inference ======")
    all_metrics = infer(model, all_features, all_case_params, infer_steps)
    dump_json(all_metrics, output_dir / "multistep_metrics.json")
    plot_metrics(all_metrics, output_dir / "multistep_metrics.pdf")


if __name__ == "__main__":
    main()
