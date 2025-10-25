from pathlib import Path
from typing import List
import time
from shutil import copyfile
from copy import deepcopy

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from dataset.base import CfdAutoDataset
from dataset import get_auto_dataset
from models.base_model import AutoCfdModel
from models.auto_deeponet import AutoDeepONet
from models.auto_edeeponet import AutoEDeepONet
from models.auto_deeponet_cnn import AutoDeepONetCnn
from models.auto_ffn import AutoFfn
from utils.common import (
    dump_json,
    plot,
    plot_loss,
    get_output_dir,
    load_best_ckpt,
    plot_predictions,
)
from utils.autoregressive import init_model
from args import Args


def collate_fn(batch: list):
    # batch is a list of tuples (input_frame, label_frame, case_params)
    inputs, labels, case_params = zip(*batch)
    inputs = torch.stack(inputs)  # (b, 3, h, w)
    labels = torch.stack(labels)  # (b, 3, h, w)

    # The last channel from features is the binary mask.
    labels = labels[:, :-1]  # (b, 2, h, w)
    mask = inputs[:, -1:]  # (b, 1, h, w)
    inputs = inputs[:, :-1]  # (b, 2, h, w)

    # Case params is a dict, turn it into a tensor
    keys = [
        x for x in case_params[0].keys() if x not in ["rotated", "dx", "dy"]
    ]
    case_params_vec = []
    for case_param in case_params:
        case_params_vec.append([case_param[k] for k in keys])
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    # Build the kwargs dict for the model's forward method
    return dict(
        inputs=inputs.cuda(),
        label=labels.cuda(),
        mask=mask.cuda(),
        case_params=case_params.cuda(),
    )


def evaluate(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    batch_size: int = 2,
    plot_interval: int = 1,
    measure_time: bool = False,
):
    if measure_time:
        assert batch_size == 1

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    scores = {name: [] for name in model.loss_fn.get_score_names()}
    input_scores = deepcopy(scores)
    all_preds: List[Tensor] = []
    print("=== Evaluating ===")
    print(f"# examples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"# batches: {len(loader)}")
    print(f"Plot interval: {plot_interval}")
    print(f"Output dir: {output_dir}")
    start_time = time.time()
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(tqdm(loader)):
            # inputs, labels, case_params = batch
            inputs = batch["inputs"]  # (b, 2, h, w)
            labels = batch["label"]  # (b, 2, h, w)

            # Compute difference between the input and label
            input_loss: dict = model.loss_fn(
                labels=labels[:, :1], preds=inputs[:, :1]
            )
            for key in input_scores:
                input_scores[key].append(input_loss[key].cpu().tolist())

            # Compute the prediction and its loss
            outputs: dict = model(**batch)
            loss: dict = outputs["loss"]
            preds: Tensor = outputs["preds"]
            height, width = labels.shape[2:]

            # When using DeepONetAuto, the prediction is a flattened.
            preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
            # loss = model.loss_fn(labels=labels[:, :1], preds=preds)
            for key in scores:
                scores[key].append(loss[key].cpu().tolist())
            # preds = preds.repeat(1, 3, 1, 1)
            all_preds.append(preds.cpu().detach())
            if step % plot_interval == 0 and not measure_time:
                # Dump input, label and prediction flow images.
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=inputs[0][0],
                    label=labels[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    if measure_time:
        print("Memory usage:")
        print(torch.cuda.memory_summary("cuda"))
        print("Time usage:")
        time_per_step = 1000 * (time.time() - start_time) / len(loader)
        print(f"Time (ms) per step: {time_per_step:.3f}")
        exit()

    avg_scores = {}
    for key in scores:
        mean = np.mean(scores[key])
        input_mean = np.mean(input_scores[key])
        avg_scores[key] = mean
        avg_scores[f"input_{key}"] = input_mean
        print(f"Prediction {key}: {mean}")
        print(f"     Input {key}: {input_mean}")

    plot_loss(scores["nmse"], output_dir / "loss.png")
    return dict(
        preds=torch.cat(all_preds, dim=0),
        scores=dict(
            mean=avg_scores,
            all=scores,
        ),
    )


def test(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    infer_steps: int = 200,
    plot_interval: int = 10,
    batch_size: int = 1,
    measure_time: bool = False,
):
    assert infer_steps > 0
    assert plot_interval > 0
    output_dir.mkdir(exist_ok=True, parents=True)
    print("=== Testing ===")
    print(f"batch_size: {batch_size}")
    print(f"Plot interval: {plot_interval}")
    result = evaluate(
        model,
        data,
        output_dir=output_dir,
        batch_size=batch_size,
        plot_interval=plot_interval,
        measure_time=measure_time,
    )
    preds = result["preds"]
    scores = result["scores"]
    torch.save(preds, output_dir / "preds.pt")
    dump_json(scores, output_dir / "scores.json")
    print("=== Testing done ===")


def train(
    model: AutoCfdModel,
    train_data: CfdAutoDataset,
    dev_data: CfdAutoDataset,
    output_dir: Path,
    num_epochs: int = 400,
    lr: float = 1e-3,
    lr_step_size: int = 1,
    lr_gamma: float = 0.9,
    batch_size: int = 2,
    eval_batch_size: int = 2,
    log_interval: int = 10,
    eval_interval: int = 2,
    measure_time: bool = False,
):
    """
    Main function for training.

    ### Parameters
    - model
    - train_data
    - dev_data
    - output_dir
    ...
    - log_interval: log loss, learning rate etc. every `log_interval` steps.
    - measure_time: if `True`, will only run one epoch and print the time.
    """
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    print("====== Training ======")
    print(f"# batch: {batch_size}")
    print(f"# examples: {len(train_data)}")
    print(f"# step: {len(train_loader)}")
    print(f"# epoch: {num_epochs}")

    start_time = time.time()
    global_step = 0
    train_losses = []

    for ep in range(num_epochs):
        ep_start_time = time.time()
        ep_train_losses = []
        for step, batch in enumerate(train_loader):
            # Forward
            outputs: dict = model(**batch)
            if step == 0 and not measure_time:
                out_file = Path("example.png")
                inputs = batch["inputs"]
                labels = batch["label"]
                preds = outputs["preds"]
                if any(
                    isinstance(model, t)
                    for t in [
                        AutoDeepONet,
                        AutoEDeepONet,
                        AutoFfn,
                        AutoDeepONetCnn,
                    ]
                ):
                    plot(inputs[0][0], labels[0][0], labels[0][0], out_file)
                else:
                    plot(inputs[0][0], labels[0][0], preds[0][0], out_file)

            # Backward
            loss: dict = outputs["loss"]
            # print(loss)
            loss["nmse"].backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            ep_train_losses.append(loss["nmse"].item())
            global_step += 1
            if global_step % log_interval == 0:
                log_info = dict(
                    ep=ep,
                    step=step,
                    mse=f"{loss['mse'].item():.3e}",
                    nmse=f"{loss['nmse'].item():.3e}",
                    lr=f"{scheduler.get_last_lr()[0]:.3e}",
                    time=round(time.time() - start_time),
                )
                print(log_info)

        if measure_time:
            print("Memory usage:")
            print(torch.cuda.memory_summary("cuda"))
            print("Time usage:")
            print(time.time() - ep_start_time)
            exit()

        scheduler.step()
        train_losses += ep_train_losses

        # Plot
        if (ep + 1) % eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            result = evaluate(
                model, dev_data, ckpt_dir, batch_size=eval_batch_size
            )
            dev_scores = result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save checkpoint
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            if ckpt_path.exists():
                ckpt_backup_path = ckpt_dir / "backup_model.pt"
                print(f"Backing up old checkpoint to {ckpt_backup_path}")
                copyfile(ckpt_path, ckpt_backup_path)
            torch.save(model.state_dict(), ckpt_path)

            # Save average scores
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["all"]["nmse"]),  # type: ignore
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")
    print("====== Training done ======")
    dump_json(train_losses, output_dir / "train_losses.json")
    plot_loss(train_losses, output_dir / "train_losses.png")


def main():
    args = Args().parse_args()
    print("#" * 80)
    print(args)
    print("#" * 80)

    output_dir = get_output_dir(args, is_auto=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json"))

    # Data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, dev_data, test_data = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    assert train_data is not None
    assert dev_data is not None
    assert test_data is not None
    print(f"# train examples: {len(train_data)}")
    print(f"# dev examples: {len(dev_data)}")
    print(f"# test examples: {len(test_data)}")

    # Model
    print("Loading model")
    model = init_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")

    if "train" in args.mode:
        args.save(str(output_dir / "train_args.json"))
        train(
            model,
            train_data=train_data,
            dev_data=dev_data,
            output_dir=output_dir,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
        )
    if "test" in args.mode:
        args.save(str(output_dir / "test_args.json"))
        # Test
        load_best_ckpt(model, output_dir)
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        test(
            model,
            test_data,
            output_dir / "test",
            batch_size=1,
            infer_steps=20,
            plot_interval=10,
        )


if __name__ == "__main__":
    main()
