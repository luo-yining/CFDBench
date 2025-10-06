import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import gc
import os

# Set memory allocation configuration BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import for Automatic Mixed Precision (AMP)
from torch.amp import autocast
from torch.amp import GradScaler 
# -----------------------------------------------------------

# Imports from the CFDBench project
from models.ldm2 import LatentDiffusionCfdModel2, LatentDiffusionCfdModelLite
from dataset import get_auto_dataset
from utils.autoregressive import init_model
from utils.common import plot_predictions, dump_json, load_best_ckpt
from args import Args

def evaluate_ldm(model: LatentDiffusionCfdModel2, dataloader, device, output_dir: Path, plot_interval: int = 50, max_eval_batches: int = 50):
    """
    Custom evaluation function for the Latent Diffusion Model.
    The mask is used here to ensure the loss is only calculated in valid fluid regions.
    Limited to max_eval_batches to reduce memory usage.
    """
    model.eval()
    total_nmse = 0.0
    num_batches = 0
    
    print("=== Evaluating LDM ===")
    progress_bar = tqdm(dataloader, desc="Evaluation", total=min(len(dataloader), max_eval_batches))
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            if i >= max_eval_batches:
                break
                
            # The collate_fn now provides the mask as a separate item
            inputs, label, case_params, mask = batch
            inputs, label, case_params, mask = inputs.to(device), label.to(device), case_params.to(device), mask.to(device)

            # Debug shapes on first iteration
            if i == 0:
                print(f"Debug shapes - inputs: {inputs.shape}, label: {label.shape}, mask: {mask.shape}")

            # Use mixed precision for evaluation too
            with autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                # The generate function only receives the 2-channel velocity input
                generated_frame = model.generate(inputs=inputs, case_params=case_params)

            # Ensure mask has proper shape - should be [B, 1, H, W] or broadcastable
            # If mask has multiple channels, use only the first one (they're all the same)
            if mask.dim() == 4 and mask.shape[1] > 1:
                mask = mask[:, 0:1, :, :]  # Keep only first channel as [B, 1, H, W]

            # Apply the mask to both the prediction and the label for a fair comparison
            # Broadcasting will automatically expand mask from [B, 1, H, W] to match [B, 2, H, W]
            loss = F.mse_loss(generated_frame * mask, label * mask)
            nmse = loss / (torch.square(label * mask).mean() + 1e-8)
            total_nmse += nmse.item()
            num_batches += 1
            
            if i % plot_interval == 0 and len(inputs) > 0:
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=inputs[0, 0].cpu(), # Plot u-velocity
                    label=label[0, 0].cpu(),
                    pred=generated_frame[0, 0].cpu(),
                    out_dir=image_dir,
                    step=i,
                )
            
            # Clear tensors immediately
            del inputs, label, case_params, mask, generated_frame, loss, nmse

    avg_nmse = total_nmse / num_batches
    print(f"Evaluation NMSE (on {num_batches} batches): {avg_nmse:.6f}")
    return {"mean": {"nmse": avg_nmse}}

def train_ldm(model: LatentDiffusionCfdModel2, train_loader, dev_loader, args, device):
    """ Main training loop for the Latent Diffusion Model. """
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize the Gradient Scaler for Automatic Mixed Precision
    scaler = GradScaler(enabled=args.use_mixed_precision)
    
    # Implement gradient accumulation
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 4)
    print(f"✓ Using gradient accumulation with {gradient_accumulation_steps} steps")
    print(f"✓ Effective batch size: {args.batch_size * gradient_accumulation_steps}")
    
    print("====== Training LDM ======")
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Mask is available from collate_fn but not used in the training forward pass
            inputs, label, case_params, _ = batch
            inputs, label, case_params = inputs.to(device), label.to(device), case_params.to(device)
            
            # We specify the device type and target dtype explicitly.
            with autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_mixed_precision):
                outputs = model(inputs=inputs, label=label, case_params=case_params)
                loss = outputs["loss"]["mse"]

                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            # --------------------------------------------

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                del inputs, label, case_params, outputs, loss
                continue

            # Use the scaler for the backward pass
            scaler.scale(loss).backward()
            
            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(loss=f"{loss.item() * gradient_accumulation_steps:.6f}")

            # Clear intermediates
            del inputs, label, case_params, outputs, loss

        # Clear cache and run garbage collection before evaluation
        torch.cuda.empty_cache()
        gc.collect()
        
        # --- Evaluation ---
        if (epoch + 1) % args.eval_interval == 0:
            output_dir = Path(args.output_dir) / "ldm_training"
            ckpt_dir = output_dir / f"ckpt-{epoch}"
            
            dev_scores = evaluate_ldm(model, dev_loader, device, ckpt_dir, max_eval_batches=50)
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            
            # Save the U-Net checkpoint
            model.unet.save_pretrained(ckpt_dir / "unet")
            print(f"Saved U-Net checkpoint to {ckpt_dir / 'unet'}")
            
            # Clear cache after saving
            torch.cuda.empty_cache()
            gc.collect()

def main():
    """Main function to train the Latent Diffusion Model."""
    args = Args().parse_args()
    args.model = "latent_diffusion2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print initial GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

   
    #rgs.batch_size = 4
    print(f"batch size: {args.batch_size}")
    
    # Use gradient accumulation to compensate for small batch sizes
    args.gradient_accumulation_steps = 1
    print(f"✓ Gradient accumulation steps: {args.gradient_accumulation_steps}")

    # --- 1. Load Data ---
    print("Loading raw dataset for LDM training...")
    train_data_raw, dev_data_raw, _ = get_auto_dataset(
        data_dir=Path(args.data_dir), data_name=args.data_name, delta_time=args.delta_time,
        norm_props=True, norm_bc=True, load_splits=['train', 'dev']
    )
    
    # --- 2. Initialize the LDM ---
    print(f"Using hardcoded scaling factor from args: {args.ldm_scaling_factor}")
    args.in_chan = 2 # Conditioning on u, v only
    
    # Set memory-efficient U-Net parameters
    if not hasattr(args, 'unet_base_channels'):
        args.unet_base_channels = 64  # Start small
    if not hasattr(args, 'unet_channel_mult'):
        args.unet_channel_mult = (1, 2, 4)  # 3 levels instead of 4
    if not hasattr(args, 'unet_num_res_blocks'):
        args.unet_num_res_blocks = 1  # Minimal residual blocks
    if not hasattr(args, 'unet_attention_resolutions'):
        args.unet_attention_resolutions = ()  # No attention layers initially
    
    print("\n" + "="*50)
    print("U-Net Architecture Settings:")
    print(f"  • Base channels: {args.unet_base_channels}")
    print(f"  • Channel multipliers: {args.unet_channel_mult}")
    print(f"  • Residual blocks per level: {args.unet_num_res_blocks}")
    print(f"  • Attention resolutions: {args.unet_attention_resolutions or 'None'}")
    print("="*50 + "\n")
    
    print("Initializing model...")
    model = init_model(args)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    model.to(device)
    
    # Print memory after model loading
    if torch.cuda.is_available():
        print(f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # --- 3. Create Final DataLoaders ---
    def collate_fn_ldm(batch):
        """
        Collate function for LDM training.

        Returns:
            input_velocities: [B, 2, H, W] - u and v velocity fields
            output_velocities: [B, 2, H, W] - target u and v velocity fields
            case_params: [B, n_params] - case parameters tensor
            mask: [B, 1, H, W] - mask for valid regions
        """
        inputs_list, labels_list, case_params_list = zip(*batch)

        inputs = torch.stack(inputs_list)
        labels = torch.stack(labels_list)

        # Separate the mask from the input velocities
        input_velocities = inputs[:, :2]  # Channels 0, 1 are u, v
        mask = inputs[:, 2:]             # Channel 2 is the mask

        # The label should only be the velocities
        output_velocities = labels[:, :2]

        keys = case_params_list[0].keys()
        case_params = torch.tensor([[d[k] for k in keys] for d in case_params_list])

        # Return the mask as a separate item
        return input_velocities, output_velocities, case_params, mask

    # Minimal workers and no pin_memory to reduce overhead
    train_loader = DataLoader(train_data_raw, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn_ldm, num_workers=4,
                              persistent_workers=True,
                              pin_memory=True)
    
    dev_loader = DataLoader(dev_data_raw, batch_size=1, 
                           shuffle=False, collate_fn=collate_fn_ldm, num_workers=2,
                           persistent_workers=True, 
                           pin_memory=True)

    print("\n" + "="*50)
    print("Memory-Optimized Configuration:")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  • Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  • Mixed precision: {args.use_mixed_precision}")
    print(f"  • Eval batch size: 1")
    print(f"  • Max eval batches: 50")
    print("="*50 + "\n")

    # --- 4. Start Training ---
    try:
        train_ldm(model, train_loader, dev_loader, args, device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "!"*50)
            print("STILL OUT OF MEMORY!")
        raise

if __name__ == "__main__":
    main()