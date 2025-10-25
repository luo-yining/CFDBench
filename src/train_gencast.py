import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
# Use Cosine Annealing scheduler
from transformers import get_cosine_schedule_with_warmup
# Using ReduceLROnPlateau as a fallback or alternative
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch version-compatible import for AMP
try:
    from torch.amp import autocast, GradScaler
    print("Using torch.amp")
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("Using torch.cuda.amp")
    except ImportError:
        # Create dummy classes if AMP is not available (CPU training)
        print("AMP not available, running in FP32.")
        class autocast:
            def __init__(self, device_type=None, dtype=None, enabled=True): pass
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        class GradScaler:
            def __init__(self, enabled=True): self.enabled=enabled
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, state_dict): pass


from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import numpy as np
import time
import gc
import os
import json

# --- CFDBench Imports ---
from dataset import get_auto_dataset
# Import the wrapper dataset
from dataset.wrapper import GenCastWrapperDataset
from models.base_model import AutoCfdModel
# Import the GenCast-style model
from models.gen_cast_cfd import GenCastCfdModel # Assuming you saved it here
from models.loss import MseLoss, loss_name_to_fn # Make sure MseLoss is imported
from utils.autoregressive import get_input_shapes # Re-use shape calculation
from utils.common import dump_json, plot_predictions, get_output_dir, load_json
from args import Args # Use the existing Args class

# --- Constants ---
RESIDUAL_STATS_FILENAME = "residual_stats.pt"

# --- Collate Function (Specific for GenCastWrapperDataset) ---
def collate_fn_gen_cast(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for the GenCastWrapperDataset.
    Stacks tensors and converts case_params dict to tensor.
    Separates velocity fields and masks.
    """
    # Stack the tensors for inputs_prev, inputs, label directly from dict items
    inputs_prev = torch.stack([item['inputs_prev'] for item in batch]) # Shape: [B, 3, H, W]
    inputs      = torch.stack([item['inputs'] for item in batch])      # Shape: [B, 3, H, W]
    labels      = torch.stack([item['label'] for item in batch])       # Shape: [B, 3, H, W]

    # Assume mask is always the last channel (-1:)
    # Separate masks and keep velocity fields (first 2 channels, :-1)
    mask_prev   = inputs_prev[:, -1:, :, :]
    mask        = inputs[:, -1:, :, :]
    mask_label  = labels[:, -1:, :, :] # Mask from the label timestep (might be useful)

    inputs_prev_vel = inputs_prev[:, :-1, :, :]
    inputs_vel      = inputs[:, :-1, :, :]
    labels_vel      = labels[:, :-1, :, :]

    # Convert case_params dict to tensor (copied from train_auto_v2.py logic)
    case_params_list = [item['case_params'] for item in batch]
    # Ensure all dicts have the same keys, filter non-numeric if needed
    keys = sorted([k for k in case_params_list[0].keys() if k not in ["rotated", "dx", "dy", "file"]])
    case_params = torch.tensor([[float(cp[k]) for k in keys] for cp in case_params_list], dtype=torch.float32)

    return {
        "inputs_prev": inputs_prev_vel, # X_{t-2} velocities [B, 2, H, W]
        "inputs":      inputs_vel,      # X_{t-1} velocities [B, 2, H, W]
        "label":       labels_vel,      # X_{t}   velocities [B, 2, H, W]
        "mask":        mask,            # Mask from X_{t-1}    [B, 1, H, W]
        "case_params": case_params,     # Case parameters tensor [B, N_params]
    }

# --- Model Initialization (Specific for GenCastCfdModel) ---
def init_gen_cast_model(args: Args, residual_mean: torch.Tensor, residual_std: torch.Tensor) -> GenCastCfdModel:
    """Instantiates the GenCast-style model."""
    loss_fn = loss_name_to_fn(args.loss_name)
    _, _, n_case_params = get_input_shapes(args) # Reuse shape calculation

    # Add GenCast specific args to Args class or use defaults here
    base_channels = getattr(args, 'gencast_base_channels', 64)
    channel_mults = getattr(args, 'gencast_channel_mults', (1, 2, 4))
    num_res_blocks = getattr(args, 'gencast_num_res_blocks', 2)
    dropout = getattr(args, 'gencast_dropout', 0.1)

    print("--- Initializing GenCastCfdModel ---")
    print(f"  Base Channels: {base_channels}")
    print(f"  Channel Multipliers: {channel_mults}")
    print(f"  Residual Blocks: {num_res_blocks}")
    print(f"  Dropout: {dropout}")
    print(f"  Input Channels (in_chan): {args.in_chan}")
    print(f"  Output Channels (out_chan): {args.out_chan}")
    print(f"  Num Case Params: {n_case_params}")
    print(f"  Residual Mean shape: {residual_mean.shape}")
    print(f"  Residual Std shape: {residual_std.shape}")

    model = GenCastCfdModel(
        in_chan=args.in_chan, # Should be 2 (u, v)
        out_chan=args.out_chan, # Should be 2 (u, v)
        loss_fn=loss_fn,
        n_case_params=n_case_params,
        residual_mean=residual_mean,
        residual_std=residual_std,
        image_size=args.num_rows if "cavity" in args.data_name else args.num_rows + 2, # Adjust based on dataset padding
        noise_scheduler_timesteps=getattr(args, 'diffusion_timesteps', 1000),
        use_gradient_checkpointing=getattr(args, 'use_gradient_checkpointing', True),
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    )
    return model


# --- Evaluation Function (Adapted for GenCast Model) ---
@torch.inference_mode() # More efficient than torch.no_grad()
def evaluate(
    model: GenCastCfdModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    plot_interval: int = 50,
    max_eval_batches: Optional[int] = 100, # Limit evaluation batches for speed/memory
    use_mixed_precision: bool = True,
) -> Dict:
    """ Evaluate the GenCast-style model. """
    model.eval()
    output_dir.mkdir(exist_ok=True, parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True, parents=True)

    score_names = model.loss_fn.get_score_names()
    scores = {name: [] for name in score_names}
    # Keep baseline scores for comparison
    input_scores = {f"input_{name}": [] for name in score_names}

    print("=== Evaluating GenCast Model ===")
    print(f"  Max batches: {max_eval_batches or 'all'}")

    total_batches = len(dataloader) if max_eval_batches is None else min(len(dataloader), max_eval_batches)
    progress_bar = tqdm(dataloader, desc="Evaluation", total=total_batches, leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch["inputs"]         # X_{t-1}
        inputs_prev = batch["inputs_prev"] # X_{t-2}
        labels = batch["label"]         # X_{t}
        mask = batch["mask"]
        case_params = batch["case_params"]

        # Compute baseline (input t-1 vs label t) scores (masked)
        baseline_loss = model.loss_fn(labels=labels * mask, preds=inputs * mask)
        for key in score_names:
            input_scores[f"input_{key}"].append(baseline_loss[key].cpu().item())

        # Forward pass using model.generate() for diffusion models during eval
        # Or model.forward() if generate is too slow / not needed for validation metric
        eval_start_time = time.time()
        # Use AMP context manager for potential speedup in evaluation too
        # Need to cast dtype explicitly inside autocast if using torch.cuda.amp
        amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_mixed_precision):
            # Pass all required arguments to forward
            outputs = model(
                inputs=inputs,
                inputs_prev=inputs_prev,
                label=labels, # Pass label to compute loss internally
                case_params=case_params,
                mask=mask
            )
            # Diffusion models typically predict noise ('preds'), not the final frame.
            # Loss is calculated on the noise prediction.
            loss = outputs["loss"]

            # To get the actual predicted frame for plotting/comparison,
            # we need to run the generation process. This might be slow.
            # Alternatively, for a quicker validation check, we could just evaluate
            # the noise prediction loss, which is what 'loss' contains.

            # For simplicity in validation, we use the internal loss calculation.
            # If you need to evaluate the *generated frame*, you'd call model.generate()
            # and compute loss manually:
            # preds_frame = model.generate(inputs=inputs, inputs_prev=inputs_prev, ...)
            # frame_loss = model.loss_fn(labels=labels * mask, preds=preds_frame * mask)


        eval_time = time.time() - eval_start_time

        # Record scores from the internal loss calculation
        for key in score_names:
            scores[key].append(loss[key].cpu().item())

        # Plot predictions (using inputs and labels for context)
        # Note: preds here is noise prediction, not the final frame.
        # Plotting noise might be less intuitive. Consider adding generation for viz.
        if batch_idx % plot_interval == 0 and len(inputs) > 0:
             # Option 1: Plot inputs/labels (simple)
             plot_predictions(
                 inp=inputs[0, 0].float().cpu(),  # u-velocity t-1
                 label=labels[0, 0].float().cpu(), # u-velocity t
                 pred=inputs_prev[0, 0].float().cpu(), # u-velocity t-2 as 'pred' for viz
                 out_dir=image_dir,
                 step=batch_idx,
             )
             # Option 2: Generate the actual frame for plotting (slower)
             # with torch.no_grad():
             #    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_mixed_precision):
             #        generated_frame = model.generate(inputs=inputs[0:1], inputs_prev=inputs_prev[0:1], case_params=case_params[0:1], mask=mask[0:1])
             # plot_predictions(
             #     inp=inputs[0, 0].float().cpu(),
             #     label=labels[0, 0].float().cpu(),
             #     pred=generated_frame[0, 0].float().cpu(),
             #     out_dir=image_dir,
             #     step=batch_idx
             # )

        # Update progress bar
        progress_bar.set_postfix({
            "nmse": f"{loss['nmse']:.4f}",
            "eval_time": f"{eval_time:.3f}s"
        })

        # Clear memory
        del batch, inputs, inputs_prev, labels, mask, case_params, outputs, loss, baseline_loss

    # Compute average scores
    avg_scores = {}
    for key in score_names:
        avg_scores[key] = float(np.mean(scores[key]))
        avg_scores[f"input_{key}"] = float(np.mean(input_scores[f"input_{key}"]))

    print("\n=== Evaluation Results ===")
    for key in score_names:
        print(f"  {key.upper():<6}: Pred={avg_scores[key]:.6f}, Input={avg_scores[f'input_{key}']:.6f}")

    return {
        "mean": avg_scores,
        "all": scores, # Keep all scores if needed for detailed analysis
    }


# --- Training Function (Adapted for GenCast Model) ---
def train(
    model: GenCastCfdModel,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    args: Args,
    device: torch.device,
    output_dir: Path,
) -> None:
    """ Main training loop for the GenCast-style model. """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(getattr(args, 'adam_beta1', 0.9), getattr(args, 'adam_beta2', 0.999)),
        eps=getattr(args, 'adam_epsilon', 1e-8),
        weight_decay=getattr(args, 'weight_decay', 1e-4), # Use weight decay from args
    )

    # Scheduler (Cosine Annealing with Warmup)
    num_training_steps = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    num_warmup_steps = getattr(args, 'lr_warmup_steps', 500)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Mixed precision scaler
    use_amp = args.use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16 # Autodetect dtype

    # Gradient accumulation
    gradient_accumulation_steps = args.gradient_accumulation_steps

    print("\n" + "=" * 40)
    print("====== Training GenCast Model ======")
    print(f"  Device: {device}")
    print(f"  AMP Enabled: {use_amp}, dtype: {amp_dtype}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad Accumulation: {gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {args.batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: Cosine w/ Warmup ({num_warmup_steps} steps)")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Total Optimization Steps: {num_training_steps}")
    print(f"  Output Dir: {output_dir}")
    print("=" * 40 + "\n")

    # Training state
    global_step = 0
    best_dev_loss = float('inf')
    start_epoch = 0
    all_train_losses_step = [] # Log loss per step

    # --- Resume from Checkpoint ---
    resume_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    if resume_checkpoint:
        ckpt_path = Path(resume_checkpoint)
        if ckpt_path.exists():
            print(f"Resuming training from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            best_dev_loss = checkpoint['best_dev_loss']
            print(f"Resumed from Epoch {start_epoch}, Step {global_step}, Best Loss {best_dev_loss:.6f}")
        else:
            print(f"WARNING: Checkpoint path not found: {ckpt_path}. Starting from scratch.")


    # --- Training Loop ---
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_losses = []
        epoch_start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)

        optimizer.zero_grad(set_to_none=True) # Reset gradients at the start of epoch

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Ensure all required inputs are passed
                outputs = model(
                    inputs=batch["inputs"],
                    inputs_prev=batch["inputs_prev"],
                    label=batch["label"], # Needed for loss calculation inside forward
                    case_params=batch["case_params"],
                    mask=batch["mask"]
                )
                # Use MSE loss on noise prediction as primary training objective
                loss = outputs["loss"]["mse"]
                loss = loss / gradient_accumulation_steps

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: Invalid loss (NaN/Inf) at epoch {epoch+1}, step {batch_idx}. Skipping update.")
                # Skip the backward pass and optimizer step for this batch
                del batch, outputs, loss
                # Need to reset gradients if accumulation was in progress
                if (batch_idx + 1) % gradient_accumulation_steps != 0:
                     optimizer.zero_grad(set_to_none=True) # Reset for the next accumulation cycle
                continue # Skip to the next batch

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient Clipping (important for stability)
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'max_grad_norm', 1.0))

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # Step scheduler after optimizer
                optimizer.zero_grad(set_to_none=True) # Reset gradients *after* update

                # Log step loss and learning rate
                actual_loss = loss.item() * gradient_accumulation_steps
                epoch_losses.append(actual_loss)
                all_train_losses_step.append(actual_loss)
                global_step += 1

                progress_bar.set_postfix({
                    "loss": f"{actual_loss:.6f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "grad_norm": f"{grad_norm:.3f}",
                })

                # Detailed Log periodically
                if global_step % args.log_interval == 0:
                    avg_recent_loss = np.mean(epoch_losses[-args.log_interval:]) if epoch_losses else 0.0
                    print(f"  [E:{epoch+1}, S:{global_step}] Loss: {avg_recent_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}, Grad Norm: {grad_norm:.3f}")

            # Clear memory
            del batch, outputs, loss

        # --- End of Epoch ---
        epoch_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_time = time.time() - epoch_start_time

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"  Avg Train Loss (MSE on noise): {epoch_train_loss:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.2e}")

        # Clear cache before evaluation
        torch.cuda.empty_cache()
        gc.collect()

        # --- Evaluation ---
        if (epoch + 1) % args.eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-epoch-{epoch}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)

            dev_scores_dict = evaluate(
                model=model,
                dataloader=dev_loader,
                device=device,
                output_dir=ckpt_dir,
                plot_interval=args.log_interval * 2, # Plot less freq during eval
                max_eval_batches=getattr(args, 'max_eval_batches', 100), # Limit eval batches
                use_mixed_precision=use_amp,
            )
            # Use NMSE for deciding best model, but MSE was used for training noise pred
            dev_loss_metric = dev_scores_dict["mean"]["nmse"] # Track NMSE on reconstructed frames

            # Save checkpoint (including optimizer, scheduler, scaler states)
            ckpt_path = ckpt_dir / "training_state.pt"
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss_epoch_avg': epoch_train_loss,
                'dev_loss_metric': dev_loss_metric, # Save the primary metric
                'best_dev_loss': best_dev_loss, # Keep track of best loss so far
                'args': vars(args) # Save args for reference
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

            # Save best model based on dev loss metric (NMSE)
            if dev_loss_metric < best_dev_loss:
                best_dev_loss = dev_loss_metric
                best_model_path = output_dir / "best_model.pt"
                # Save only the model state dict for the best model
                torch.save(model.state_dict(), best_model_path)
                print(f"  *** New best model saved! Dev NMSE: {dev_loss_metric:.6f} ***")

            # Save scores
            dump_json(dev_scores_dict, ckpt_dir / "dev_scores.json")
            dump_json({"epoch": epoch, "train_loss_avg": epoch_train_loss}, ckpt_dir / "train_epoch_summary.json")

            # Clear cache after evaluation
            torch.cuda.empty_cache()
            gc.collect()

    # --- End of Training ---
    print("\n" + "=" * 40)
    print("====== Training Complete ======")
    print(f"  Best dev NMSE: {best_dev_loss:.6f}")
    print(f"  Best model saved to: {output_dir / 'best_model.pt'}")
    # Save final training losses per step
    dump_json(all_train_losses_step, output_dir / "all_train_losses_step.json")
    print("=" * 40 + "\n")


# --- Main Execution ---
def main():
    """Main entry point for training."""
    args = Args().parse_args()

    # --- Setup ---
    print("\n" + "=" * 80)
    print("CFDBench GenCast-Style Model Training")
    print("=" * 80 + "\n")
    print(args)
    print("\n" + "=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Set memory config - might help fragmentation but requires PyTorch >= 1.10
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Initial Reserved:   {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")

    output_dir = get_output_dir(args, is_auto=True) / "gencast_model" # Subdirectory for this model
    output_dir.mkdir(exist_ok=True, parents=True)
    args.save(str(output_dir / "args.json")) # Save args used for this run

    # --- Load Residual Stats ---
    stats_path = Path(getattr(args, 'residual_stats_path', RESIDUAL_STATS_FILENAME))
    if not stats_path.exists():
        print(f"ERROR: Residual statistics file not found at {stats_path}")
        print("Please run `calculate_stats.py` first.")
        return
    print(f"Loading residual stats from {stats_path}")
    stats = torch.load(stats_path, map_location='cpu') # Load to CPU initially
    residual_mean = stats['residual_mean']
    residual_std = stats['residual_std']

    # --- Load Datasets ---
    print("Loading and wrapping datasets...")
    train_data_raw, dev_data_raw, test_data_raw = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time, # This delta_time determines t vs t-1 step
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    # Check if datasets loaded correctly
    if train_data_raw is None or dev_data_raw is None:
        print("Error: Failed to load training or validation data.")
        return

    # Wrap datasets
    train_data = GenCastWrapperDataset(train_data_raw)
    dev_data = GenCastWrapperDataset(dev_data_raw)
    test_data = GenCastWrapperDataset(test_data_raw) if test_data_raw else None

    print(f"  Wrapped Train: {len(train_data)} examples")
    print(f"  Wrapped Dev:   {len(dev_data)} examples")
    if test_data: print(f"  Wrapped Test:  {len(test_data)} examples\n")

    # --- Create DataLoaders ---
    # Reduce num_workers if experiencing DataLoader hanging issues
    num_workers = getattr(args, 'num_workers', 4)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_gen_cast, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    dev_loader = DataLoader(
        dev_data, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn_gen_cast, num_workers=num_workers//2, pin_memory=True, persistent_workers=num_workers//2 > 0)
    test_loader = DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_fn_gen_cast, num_workers=num_workers//2) if test_data else None

    # --- Initialize Model ---
    model = init_gen_cast_model(args, residual_mean, residual_std)
    model.to(device) # Move model (including residual stats buffers) to device

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params / 1e6:.2f}M")
    print(f"  Trainable:      {trainable_params / 1e6:.2f}M\n")
    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Reserved memory:             {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")

    # --- Train ---
    if "train" in args.mode:
        try:
            train(
                model=model,
                train_loader=train_loader,
                dev_loader=dev_loader,
                args=args,
                device=device,
                output_dir=output_dir,
            )
        except Exception as e:
            print("\n" + "!"*80)
            print("An error occurred during training:")
            print(e)
            print("!"*80 + "\n")
            # Optionally: re-raise the exception after logging
            # raise e

    # --- Test ---
    if "test" in args.mode and test_loader:
        print("\n=== Testing ===")
        test_dir = output_dir / "test_results"
        test_dir.mkdir(exist_ok=True, parents=True)

        # Load best checkpoint for testing
        best_ckpt_path = output_dir / "best_model.pt"
        if best_ckpt_path.exists():
            print(f"Loading best model from {best_ckpt_path} for testing.")
            model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        else:
            print("WARNING: Best model checkpoint not found. Testing with last model state.")

        test_scores = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            output_dir=test_dir,
            plot_interval=20, # Plot more frequently during test if needed
            max_eval_batches=None, # Evaluate on the full test set
            use_mixed_precision=args.use_mixed_precision,
        )

        dump_json(test_scores, test_dir / "test_scores.json")
        print("\n=== Testing Complete ===")
        print(f"Test results saved in: {test_dir}")

    elif "test" in args.mode and not test_loader:
         print("\nSkipping testing phase as no test data was loaded.")

if __name__ == "__main__":
    main()

'''
**How to Use:**

1.  **Save:** Save this code as `src/train_gencast.py`.
2.  **Add `GenCastCfdModel`:** Make sure your `GenCastCfdModel` class (from our previous conversation) is saved in `src/models/gen_cast_cfd.py` (or adjust the import).
3.  **Run `calculate_stats.py`:** You *must* run `calculate_stats.py` first (using the `GenCastWrapperDataset` inside it as well!) to generate the `residual_stats.pt` file.
4.  **Modify `Args`:** You might need to add specific arguments for GenCast to `src/args.py` if you didn't use the `getattr` defaults I added (e.g., `--gencast-base-channels`, `--diffusion-timesteps`).
5.  **Run Training:**
    ```bash
    python src/train_gencast.py --data_name cylinder_prop_bc_geo --batch_size 4 --eval_batch_size 2 --num_epochs 50 --lr 1e-4 --gradient_accumulation_steps 2 --use_mixed_precision True --output_dir result/auto_v2 ... [other args]
    
'''