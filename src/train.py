from pathlib import Path
from shutil import copyfile
from typing import Any, Dict
import time

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from dataset import get_dataset, CfdDataset
from models.base_model import CfdModel
from models.deeponet import DeepONet
from models.ffn import FfnModel
from models.loss import loss_name_to_fn
from utils.common import (
    dump_json,
    plot_loss,
    get_output_dir,
    load_best_ckpt,
    plot_predictions,
)
from args import Args


def collate_fn(batch: list):
    case_params, t, label = zip(*batch)
    case_params = torch.stack(case_params)  # (b, p)
    label = torch.stack(label)  # (b, c, h, w), c=2
    t = torch.stack(t)  # (b, 1)
    return dict(
        case_params=case_params.cuda(),
        t=t.cuda(),
        label=label.cuda(),
    )


def evaluate(
    model: CfdModel,
    data: CfdDataset,
    output_dir: Path,
    batch_size: int = 64,
    plot_interval: int = 1,
    measure_time: bool = False,
) -> Dict[str, Any]:
    if measure_time:
        assert batch_size == 1

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    scores = {name: [] for name in model.loss_fn.get_score_names()}
    all_preds = []
    model.eval()

    print("=== Evaluating ===")
    print(f"# examples: {len(data)}")
    print(f"batch size: {batch_size}")
    print(f"# batches: {len(loader)}")
    print(f"Plot interval: {plot_interval}")
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader)):
            case_params = batch["case_params"]  # (b, 5)
            label = batch["label"]
            t = batch["t"]  # (b, 1)

            height, width = label.shape[-2:]

            # Compute the prediction and its loss
            preds = model.generate_one(
                case_params=case_params, t=t, height=height, width=width
            )
            loss: dict = model.loss_fn(labels=label[:, :1], preds=preds)
            for key in scores:
                scores[key].append(loss[key].item())

            preds = preds.repeat(1, 3, 1, 1)
            all_preds.append(preds.cpu().detach())
            if step % plot_interval == 0 and not measure_time:
                # Dump input, label and prediction flow images.
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=None,
                    label=label[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    if measure_time:
        print("Memory usage:")
        print(torch.cuda.memory_summary("cuda"))
        print("Time usage:")
        time_per_step = 1000 * (time.time() - start_time) / len(loader)
        print(f"Time per step: {time_per_step:.3f} ms")
        exit()

    avg_scores = {key: np.mean(vals) for key, vals in scores.items()}
    for key, vals in scores.items():
        print(f"{key}: {np.mean(vals)}")

    plot_loss(scores["nmse"], output_dir / "loss.png")
    return dict(
        scores=dict(
            mean=avg_scores,
            all=scores,
        ),
        preds=all_preds,
    )


def test(
    model: CfdModel,
    data: CfdDataset,
    output_dir: Path,
    plot_interval: int = 10,
    batch_size: int = 1,
    measure_time: bool = False,
):
    """
    Perform inference on the test set.
    """
    assert plot_interval > 0
    output_dir.mkdir(exist_ok=True, parents=True)
    print("==== Testing ====")
    print(f"Batch size: {batch_size}")
    print(f"Plot interval: {plot_interval}")
    result = evaluate(
        model,
        data,
        output_dir,
        batch_size=batch_size,
        plot_interval=plot_interval,
        measure_time=measure_time,
    )
    preds = result["preds"]
    scores = result["scores"]
    torch.save(preds, output_dir / "preds.pt")
    dump_json(scores, output_dir / "scores.json")
    print("==== Testing done ====")


def train(
    model: CfdModel,
    train_data: CfdDataset,
    dev_data: CfdDataset,
    output_dir: Path,
    num_epochs: int = 400,
    lr: float = 1e-3,
    lr_step_size: int = 1,
    lr_gamma: float = 0.9,
    batch_size: int = 64,
    log_interval: int = 50,
    eval_interval: int = 2,
    measure_time: bool = False,
):
    loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=lr_step_size, gamma=lr_gamma
    )

    print("==== Training ====")
    print(f"Output dir: {output_dir}")
    print(f"# lr: {lr}")
    print(f"# batch: {batch_size}")
    print(f"# examples: {len(train_data)}")
    print(f"# step: {len(loader)}")
    print(f"# epoch: {num_epochs}")

    start_time = time.time()
    global_step = 0
    all_train_losses = []

    for ep in range(num_epochs):
        ep_start_time = time.time()
        ep_train_losses = []
        for step, batch in enumerate(loader):
            # Forward
            outputs = model(**batch)
            losses = outputs["loss"]
            loss = losses["nmse"]

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            ep_train_losses.append(loss.item())
            global_step += 1
            if global_step % log_interval == 0 and not measure_time:
                avg_loss = sum(ep_train_losses) / (len(ep_train_losses) + 1e-5)
                log_info = {
                    "ep": ep,
                    "step": step,
                    # "loss": f"{loss.item():.3e}",
                    "loss": f"{avg_loss:.3e}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                    "time": round(time.time() - start_time),
                }
                print(log_info)

        if measure_time:
            print("Memory usage:")
            print(torch.cuda.memory_summary("cuda"))
            print("Time usage:")
            print(time.time() - ep_start_time)
            exit()

        scheduler.step()

        # Evaluate
        if (ep + 1) % eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            dev_result = evaluate(model, dev_data, ckpt_dir)
            dev_scores = dev_result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_loss.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save checkpoint
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            if ckpt_path.exists():
                ckpt_backup_path = ckpt_dir / "backup_model.pt"
                copyfile(ckpt_path, ckpt_backup_path)
            torch.save(model.state_dict(), ckpt_path)

            # Save average scores
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["mean"]["nmse"]),
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")

        all_train_losses.append(ep_train_losses)

    all_train_losses = sum(all_train_losses, [])
    dump_json(all_train_losses, output_dir / "train_losses.json")
    plot_loss(all_train_losses, output_dir / "train_losses.png")


def init_model(args: Args) -> CfdModel:
    """
    Instantiate a nonautoregressive model.
    """
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


def main():
    args = Args().parse_args()
    print(args)

    output_dir = get_output_dir(args)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json"))

    # Data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, dev_data, test_data = get_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    print(f"# train examples: {len(train_data)}")
    print(f"# dev examples: {len(dev_data)}")
    print(f"# test examples: {len(test_data)}")

    # Model
    print("Loading model")
    model = init_model(args)

    if "train" in args.mode:
        args.save(str(output_dir / "train_args.json"))
        train(
            model,
            train_data,
            dev_data,
            output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            num_epochs=args.num_epochs,
            eval_interval=args.eval_interval,
        )
    if "test" in args.mode:
        args.save(str(output_dir / "test_args.json"))
        # Test
        # Load best test ckpt
        load_best_ckpt(model, output_dir)
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        test(
            model,
            data=test_data,
            output_dir=output_dir / "test",
            batch_size=1,
            plot_interval=10,
        )


if __name__ == "__main__":
    main()
