from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor, FloatTensor
from torch.nn import functional as F

from args import Args
from utils import load_best_ckpt, get_output_dir
from dataset import get_auto_dataset
from models.base_model import AutoCfdModel
from utils_auto import init_model


def get_metrics(preds: Tensor, labels: Tensor):
    mse = ((preds - labels) ** 2).mean()
    nmse = mse / ((labels**2).mean())
    mae = F.l1_loss(preds, labels)
    return dict(
        mse=mse,
        nmse=nmse,
        mae=mae,
    )


def case_params_to_tensor(case_params: dict):
    # Case params is a dict, turn it into a tensor
    keys = [x for x in case_params.keys() if x not in ["rotated", "dx", "dy"]]
    case_params_vec = [case_params[k] for k in keys]
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    return case_params


def combine_dicts(dicts: List[dict]) -> dict:
    result = {}
    print(dicts)
    for key in dicts[0]:
        result[key] = np.mean([d[key] for d in dicts])
    return result


def infer_case(
    model: AutoCfdModel, case_features: Tensor, case_params: Tensor, infer_steps: int
):
    start_frame = case_features[0]
    preds = model.generate_many(start_frame, case_params, infer_steps)
    return preds


def infer(
    model, all_features: List[Tensor], all_case_params: List[Tensor], infer_steps: int
):
    n_cases = len(all_features)
    print(n_cases)
    all_preds = []
    for case_id, (case_features, case_params) in enumerate(
        zip(all_features, all_case_params)
    ):
        case_pred = infer_case(model, case_features, case_params, infer_steps)
        all_preds.append(case_pred)

    # Compute metrics
    all_metrics = []
    for step in range(infer_steps):
        for case_id in range(n_cases):
            case_features = all_features[case_id, step]
            case_pred = all_preds[case_id, step]
            print(case_features.shape, case_pred.shape)
            exit()
            mask = case_features[:, 2]
            case_pred = case_pred * mask
            case_features = case_features[:, :2] * mask
            metrics = get_metrics(case_pred, case_features)
            all_metrics.append(metrics)
        metrics = combine_dicts(all_metrics)
        print(metrics)
    return metrics


def main():
    args = Args().parse_args()

    print(args)
    data_dir = Path(args.data_dir)
    _, _, test_data = get_auto_dataset(
        # args.data_name,
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    print(test_data.case_dirs)
    infer_steps = 10

    all_features = test_data.all_features
    all_case_params = test_data.case_params
    print(all_case_params)
    print(all_features)

    # Make sure each case has at least `infer_steps` steps by repeating
    # the last frame. This is because we assume that flow has
    # reached steady state.
    for case_id, case_features in enumerate(all_features):
        num_frames = case_features.shape[0]
        while num_frames < infer_steps:
            case_features = np.concatenate([case_features, case_features[-1:]], axis=0)
            num_frames += 1
        all_features[case_id] = FloatTensor(case_features).to('cuda')

    # Turn case params into tensors
    for case_id, case_params in enumerate(all_case_params):
        all_case_params[case_id] = case_params_to_tensor(case_params).to('cuda')

    # Load model
    model = init_model(args)
    output_dir = get_output_dir(args, is_auto=True)
    load_best_ckpt(model, output_dir)

    print("====== Start inference ======")
    infer(model, all_features, all_case_params, infer_steps)


if __name__ == "__main__":
    main()
