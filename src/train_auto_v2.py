"""
Improved training script for autoregressive CFD models.

Features:
- Mixed precision training support
- Proper handling of both flattened and 2D predictions
- Better logging and checkpointing
- Memory-efficient evaluation
- Gradient accumulation support
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch version-compatible import
try:
    from torch.amp import autocast, GradScaler
    USE_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_TORCH_AMP = False
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import numpy as np
import time
import gc
import os

from dataset import get_auto_dataset
from dataset.base import CfdAutoDataset
from models.base_model import AutoCfdModel
from models.auto_deeponet import AutoDeepONet
from models.auto_edeeponet import AutoEDeepONet
from models.auto_ffn import AutoFfn
from models.auto_deeponet_cnn import AutoDeepONetCnn
from utils.autoregressive import init_model
from utils.common import dump_json, plot_predictions, get_output_dir
from args import Args


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for autoregressive CFD datasets.

    Returns a dictionary with:
        - inputs: [B, 2, H, W] - velocity fields (u, v)
        - label: [B, 2, H, W] - target velocity fields
        - mask: [B, 1, H, W] - binary mask for valid regions
        - case_params: [B, n_params] - case parameters
    """
    inputs_list, labels_list, case_params_list = zip(*batch)

    inputs = torch.stack(inputs_list)  # (B, 3, H, W) - includes mask
    labels = torch.stack(labels_list)  # (B, 3, H, W)

    # Separate mask from velocity channels
    mask = inputs[:, -1:, :, :]  # (B, 1, H, W)
    inputs = inputs[:, :-1, :, :]  # (B, 2, H, W)
    labels = labels[:, :-1, :, :]  # (B, 2, H, W)

    # Convert case_params dict to tensor
    keys = [k for k in case_params_list[0].keys() if k not in ["rotated", "dx", "dy"]]
    case_params = torch.tensor([[cp[k] for k in keys] for cp in case_params_list])

    return {
        "inputs": inputs,
        "label": labels,
        "mask": mask,
        "case_params": case_params,
    }


def is_flattened_output_model(model: AutoCfdModel) -> bool:
    """
    Check if the model outputs flattened predictions (e.g., DeepONet variants).
    """
    return isinstance(model, (AutoDeepONet, AutoEDeepONet, AutoFfn, AutoDeepONetCnn))


def reshape_predictions(preds: torch.Tensor, model: AutoCfdModel, target_shape: tuple) -> torch.Tensor:
    """
    Reshape predictions to match target shape.

    Args:
        preds: Model predictions (may be flattened or 2D)
        model: The model instance
        target_shape: Target shape (B, C, H, W)

    Returns:
        Reshaped predictions with shape (B, C, H, W)
    """
    batch_size, channels, height, width = target_shape

    if is_flattened_output_model(model):
        # DeepONet variants output flattened (B, H*W) or (B, 1)
        # Reshape to (B, 1, H, W) for single channel output
        if preds.dim() == 2:
            return preds.view(batch_size, 1, height, width)
        elif preds.dim() == 4 and preds.shape[1] == 1:
            return preds

    # CNN-based models already output (B, C, H, W)
    return preds


@torch.no_grad()
def evaluate(
    model: AutoCfdModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Optional[Path] = None,
    plot_interval: Optional[int] = 50,
    max_eval_batches: Optional[int] = None,
    use_mixed_precision: bool = True,
) -> Dict:
    """
    Evaluate the model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results (None = don't save anything)
        plot_interval: Plot predictions every N batches (None = skip all image saving)
        max_eval_batches: Maximum number of batches to evaluate (None = all)
        use_mixed_precision: Whether to use mixed precision

    Returns:
        Dictionary with evaluation scores
    """
    model.eval()

    # Only create directories if we're actually saving outputs
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True, parents=True)
    else:
        image_dir = None

    score_names = model.loss_fn.get_score_names()
    scores = {name: [] for name in score_names}
    input_scores = {f"input_{name}": [] for name in score_names}

    print("=== Evaluating ===")
    print(f"  Output dir: {output_dir}")
    print(f"  Plot interval: {plot_interval}")
    print(f"  Max batches: {max_eval_batches or 'all'}")

    total_batches = len(dataloader) if max_eval_batches is None else min(len(dataloader), max_eval_batches)
    progress_bar = tqdm(dataloader, desc="Evaluation", total=total_batches)

    for batch_idx, batch in enumerate(progress_bar):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch["inputs"]
        labels = batch["label"]
        mask = batch["mask"]

        # Compute baseline (input vs label) scores
        # Using inputs as predictions to see how well a naive "no change" model would do
        baseline_loss = model.loss_fn(preds=inputs, labels=labels)
        for key in score_names:
            input_scores[f"input_{key}"].append(baseline_loss[key].cpu().item())

        # Forward pass with mixed precision
        if USE_TORCH_AMP:
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_mixed_precision):
                outputs = model(**batch)
        else:
            with autocast(enabled=use_mixed_precision):
                outputs = model(**batch)

        loss = outputs["loss"]
        preds = outputs["preds"]

        # Reshape predictions if needed
        preds = reshape_predictions(preds, model, labels.shape)

        # Compute all metrics using the loss function
        # This ensures we get all metrics that get_score_names() promised
        all_metrics = model.loss_fn(preds, labels)

        # Record scores
        for key in score_names:
            scores[key].append(all_metrics[key].cpu().item())

        # Plot predictions (only if we have an output directory and plot_interval is set)
        if image_dir is not None and plot_interval is not None and batch_idx % plot_interval == 0 and len(inputs) > 0:
            plot_predictions(
                inp=inputs[0, 0].cpu(),  # u-velocity
                label=labels[0, 0].cpu(),
                pred=preds[0, 0].cpu(),
                out_dir=image_dir,
                step=batch_idx,
            )

        # Update progress bar
        progress_bar.set_postfix({k: f"{np.mean(v):.6f}" for k, v in scores.items()})

        # Clear memory
        del batch, inputs, labels, mask, outputs, loss, preds, baseline_loss

    # Compute average scores
    avg_scores = {}
    for key in score_names:
        avg_scores[key] = float(np.mean(scores[key]))
        avg_scores[f"input_{key}"] = float(np.mean(input_scores[f"input_{key}"]))

    print("\n=== Evaluation Results ===")
    for key in score_names:
        print(f"  {key:8s}: pred={avg_scores[key]:.6f}, input={avg_scores[f'input_{key}']:.6f}")

    return {
        "mean": avg_scores,
        "all": scores,
    }


def train(
    model: AutoCfdModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    args: Args,
    device: torch.device,
    output_dir: Path,
) -> None:
    """
    Main training loop for autoregressive CFD models.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        dev_loader: DataLoader for validation data
        args: Training arguments
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
    )

    # Mixed precision scaler (only enable on CUDA devices)
    use_amp = args.use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Gradient accumulation
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    print("====== Training Configuration ======")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Mixed precision: {args.use_mixed_precision}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    print("=" * 40)

    # Training state
    global_step = 0
    best_dev_loss = float('inf')
    all_train_losses = []

    # Track epoch-level losses for plotting
    epoch_train_losses = []  # Average train loss per epoch
    epoch_dev_losses = []    # Dev loss per epoch (when evaluated)

    # Start training
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        epoch_start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if USE_TORCH_AMP:
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(**batch)
                    loss = outputs["loss"][args.loss_name]
                    loss = loss / gradient_accumulation_steps
            else:
                with autocast(enabled=use_amp):
                    outputs = model(**batch)
                    loss = outputs["loss"][args.loss_name]
                    loss = loss / gradient_accumulation_steps

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: Invalid loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                del batch, outputs, loss
                continue

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Record loss
            actual_loss = loss.item() * gradient_accumulation_steps
            epoch_losses.append(actual_loss)
            all_train_losses.append(actual_loss)
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{actual_loss:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # Clear memory
            del batch, outputs, loss

        # Epoch statistics
        epoch_train_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time

        # Track epoch-level train loss
        epoch_train_losses.append(float(epoch_train_loss))

        print(f"\n=== Epoch {epoch+1} Summary ===")
        print(f"  Train loss: {epoch_train_loss:.6f}")
        print(f"  Epoch time: {epoch_time:.1f}s")

        # Clear cache before evaluation
        torch.cuda.empty_cache()
        gc.collect()

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            # Determine what to save this epoch
            should_save_checkpoint = (epoch + 1) % args.save_checkpoint_every_n_epochs == 0
            should_save_images = (epoch + 1) % args.save_images_every_n_epochs == 0

            # Only create checkpoint directory if we're actually saving something substantial
            # Otherwise just do evaluation without creating directories
            if should_save_checkpoint or should_save_images:
                ckpt_dir = output_dir / f"ckpt-{epoch}"
                plot_interval = 50 if should_save_images else None
            else:
                # Don't save anything - just run evaluation
                ckpt_dir = None
                plot_interval = None

            dev_scores = evaluate(
                model=model,
                dataloader=dev_loader,
                device=device,
                output_dir=ckpt_dir,
                plot_interval=plot_interval,
                max_eval_batches=getattr(args, 'max_eval_batches', None),
                use_mixed_precision=use_amp,
            )

            dev_loss = dev_scores["mean"][args.loss_name]

            # Track epoch-level dev loss
            epoch_dev_losses.append({
                "epoch": epoch,
                "dev_loss": float(dev_loss)
            })

            # Update learning rate scheduler
            scheduler.step(dev_loss)

            # Save full checkpoint only every N epochs (to save disk space)
            if should_save_checkpoint:
                ckpt_dir.mkdir(exist_ok=True, parents=True)  # Create directory now
                ckpt_path = ckpt_dir / "model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'dev_loss': dev_loss,
                }, ckpt_path)
                print(f"  Full checkpoint saved to {ckpt_path}")

                # Save scores alongside the checkpoint
                epoch_summary = {
                    "epoch": epoch,
                    "train_loss": float(epoch_train_loss),
                    "dev_loss": float(dev_loss),
                    "dev_scores": dev_scores["mean"],
                    "time": float(epoch_time),
                    "lr": optimizer.param_groups[0]['lr'],
                }
                dump_json(epoch_summary, ckpt_dir / "scores.json")
                dump_json(dev_scores, ckpt_dir / "dev_scores.json")
                dump_json(epoch_losses, ckpt_dir / "train_losses.json")
            else:
                next_checkpoint_epoch = ((epoch // args.save_checkpoint_every_n_epochs) + 1) * args.save_checkpoint_every_n_epochs
                print(f"  Evaluated without saving checkpoint (next checkpoint at epoch {next_checkpoint_epoch})")

            # Save best model (always save when it improves!)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_ckpt_path = output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  *** New best model saved! Dev loss: {dev_loss:.6f} ***")

            # Clear cache after evaluation
            torch.cuda.empty_cache()
            gc.collect()

    # Save final training losses
    dump_json(all_train_losses, output_dir / "all_train_losses.json")

    # Save epoch-level loss history for plotting
    loss_history = {
        "train_losses": epoch_train_losses,  # List of average train loss per epoch
        "dev_losses": epoch_dev_losses,       # List of {epoch, dev_loss} dicts
        "epochs": list(range(args.num_epochs)),
    }
    dump_json(loss_history, output_dir / "loss_history.json")
    print(f"  Loss history saved to: {output_dir / 'loss_history.json'}")

    print("\n====== Training Complete ======")
    print(f"  Best dev loss: {best_dev_loss:.6f}")
    print(f"  Best model saved to: {output_dir / 'best_model.pt'}")


def main():
    """Main entry point for training."""
    args = Args().parse_args()

    print("\n" + "=" * 80)
    print("CFDBench Autoregressive Model Training (v2)")
    print("=" * 80)
    print(args)
    print("=" * 80 + "\n")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB\n")

    # Set memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Output directory
    output_dir = get_output_dir(args, is_auto=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json"))

    # Load datasets
    print("Loading datasets...")
    train_data, dev_data, test_data = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    print(f"  Train: {len(train_data)} examples")
    print(f"  Dev: {len(dev_data)} examples")
    print(f"  Test: {len(test_data)} examples\n")

    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Initialize model
    print("Initializing model...")
    model = init_model(args)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M\n")

    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB\n")

    # Mixed precision setting (only enable on CUDA devices)
    use_amp = args.use_mixed_precision and device.type == 'cuda'

    # Train
    if "train" in args.mode:
        train(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            args=args,
            device=device,
            output_dir=output_dir,
        )

    # Test
    if "test" in args.mode:
        print("\n=== Testing ===")
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True, parents=True)

        # Load best checkpoint
        best_ckpt_path = output_dir / "best_model.pt"
        if best_ckpt_path.exists():
            print(f"Loading best model from {best_ckpt_path}")
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        test_scores = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=test_dir,
            plot_interval=10,
            use_mixed_precision=use_amp,
        )

        dump_json(test_scores, test_dir / "test_scores.json")
        print("\n=== Testing Complete ===")


if __name__ == "__main__":
    main()
