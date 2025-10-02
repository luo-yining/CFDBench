import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import numpy as np

# --- Updated Imports ---
# 1. Import the AutoencoderKL model and the new Args class
from diffsci.models.nets.autoencoderldm2d import AutoencoderKL
from args import Args # Assuming your new Args class is in args.py

# (Assuming these are your existing project imports)
from dataset import get_auto_dataset
from dataset.vae import VaeDataset


def plot_loss_history(history, save_path):
    """
    Plots training and validation loss curves using an Exponential Moving Average (EMA)
    to smooth the noisy training loss for better visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("VAE Training Loss History (per Step)", fontsize=16)

    # --- NEW: Exponential Moving Average Helper Function ---
    def exponential_moving_average(data, beta=0.99):
        """
        Computes the exponential moving average of a time series.
        - beta: The smoothing factor. Higher beta means more smoothing.
        """
        ema_data = np.zeros_like(data, dtype=float)
        ema_data[0] = data[0]
        for i in range(1, len(data)):
            ema_data[i] = beta * ema_data[i-1] + (1 - beta) * data[i]
        return ema_data

    # --- Plot Total Loss ---
    train_steps = np.arange(len(history['train_total']))
    axes[0].plot(train_steps, history['train_total'], label='Train Loss (Raw)', alpha=0.2, color='lightblue')

    # Plot smoothed EMA training loss
    if len(history['train_total']) > 1:
        ema_train_loss = exponential_moving_average(history['train_total'])
        axes[0].plot(train_steps, ema_train_loss, label='Train Loss (Smoothed EMA)', color='blue')

    axes[0].step(history['val_steps'], history['val_total'], label='Validation Loss', where='post', color='orange', linewidth=2)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss (Log Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # --- Plot Reconstruction Loss ---
    axes[1].plot(train_steps, history['train_recon'], label='Train Recon Loss (Raw)', alpha=0.2, color='lightblue')
    if len(history['train_recon']) > 1:
        ema_train_recon = exponential_moving_average(history['train_recon'])
        axes[1].plot(train_steps, ema_train_recon, label='Train Recon Loss (Smoothed EMA)', color='blue')
    axes[1].step(history['val_steps'], history['val_recon'], label='Validation Recon Loss', where='post', color='orange', linewidth=2)
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Training Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # --- Plot KL Loss ---
    axes[2].plot(train_steps, history['train_kl'], label='Train KL Loss (Raw)', alpha=0.2, color='lightblue')
    if len(history['train_kl']) > 1:
        ema_train_kl = exponential_moving_average(history['train_kl'])
        axes[2].plot(train_steps, ema_train_kl, label='Train KL Loss (Smoothed EMA)', color='blue')
    axes[2].step(history['val_steps'], history['val_kl'], label='Validation KL Loss', where='post', color='orange', linewidth=2)
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Training Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"✅ Loss history plot saved to: {save_path}")


def main():
    """Main function to train the VAE."""
    # When running in a notebook, pass an empty list to parse_args
    args = Args().parse_args()
    print("--- Training AutoencoderKL ---")
    print(args)

    # --- 1. Load Data ---
    # This section remains the same
    print("Loading data...")
    splits_to_load = ['train', 'dev']
    problem_name = args.data_name.split("_")[0]
    subset_name = args.data_name[len(problem_name) + 1:]

    if args.clear_cache:
        for split in splits_to_load:
            cache_dir_to_delete = Path(f"./dataset/cache/{problem_name}/{subset_name}/{split}")
            if cache_dir_to_delete.exists():
                print(f"Deleting outdated cache for '{split}' split at: {cache_dir_to_delete}")
                shutil.rmtree(cache_dir_to_delete)
                print("Cache deleted.")
    
    train_data_raw, dev_data_raw, _ = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        load_splits=splits_to_load
    )
    assert train_data_raw is not None and dev_data_raw is not None

    train_dataset = VaeDataset(train_data_raw, normalize=True)
    dev_dataset = VaeDataset(dev_data_raw, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train dataset created with {len(train_dataset)} frames.")
    print(f"Validation dataset created with {len(dev_dataset)} frames.")

    history = {
        'train_total': [], 'train_recon': [], 'train_kl': [],
        'val_total': [], 'val_recon': [], 'val_kl': [],
        'val_steps': []
    }

    # --- 2. Initialize Model and Optimizer (Updated) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2a. Get configurations from the Args class
    dd_config = args.get_ddconfig()
    loss_config = args.get_lossconfig()

    # 2b. Instantiate the AutoencoderKL model
    model = AutoencoderKL(
        ddconfig=dd_config,
        lossconfig=loss_config,
        embed_dim=args.embed_dim
    ).to(device)
    
    # The optimizer now targets the parameters of the AutoencoderKL model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.vae_weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

    # --- 3. Training Loop (Updated) ---
    print("Starting training loop...")
    best_val_loss = np.inf
    patience_counter = 0
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        # We still track these for logging purposes
        total_train_loss, total_train_recon, total_train_kl = 0.0, 0.0, 0.0
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 3a. New forward pass and loss calculation
            reconstructions, posterior = model(batch)
            
            # The model's internal loss function computes everything for us
            loss, log_dict = model.loss(
                batch,
                reconstructions,
                posterior,
                0, # optimizer_idx (not used with single optimizer)
                global_step,
                last_layer=model.get_last_layer(),
                split="train"
            )
            
            loss.backward()
            optimizer.step()

            # 3b. Log metrics from the dictionary returned by the loss function
            recon_loss = log_dict["train/rec_loss"]
            kl_loss = log_dict["train/kl_loss"]
            
            history['train_total'].append(loss.item())
            history['train_recon'].append(recon_loss) # Already a float
            history['train_kl'].append(kl_loss)       # Already a float
            global_step += 1
            
            total_train_loss += loss.item()
            total_train_recon += recon_loss
            total_train_kl += kl_loss
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon = total_train_recon / len(train_loader)
        avg_train_kl = total_train_kl / len(train_loader)

        # --- Validation Step (Updated) ---
        model.eval()
        total_val_loss, total_val_recon, total_val_kl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                reconstructions, posterior = model(batch)
                
                # Use the same internal loss function for validation
                loss, log_dict = model.loss(
                    batch,
                    reconstructions,
                    posterior,
                    0,
                    global_step,
                    last_layer=model.get_last_layer(),
                    split="val"
                )
                
                # Get metrics from the log dictionary
                recon_loss = log_dict["val/rec_loss"]
                kl_loss = log_dict["val/kl_loss"]
                
                total_val_loss += loss.item()
                total_val_recon += recon_loss
                total_val_kl += kl_loss
        
        avg_val_loss = total_val_loss / len(dev_loader)
        avg_val_recon = total_val_recon / len(dev_loader)
        avg_val_kl = total_val_kl / len(dev_loader)

        history['val_total'].append(avg_val_loss)
        history['val_recon'].append(avg_val_recon)
        history['val_kl'].append(avg_val_kl)
        history['val_steps'].append(global_step)

        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train -> Total: {avg_train_loss:.6f} | Recon: {avg_train_recon:.6f} | KL: {avg_train_kl:.4f}")
        print(f"  Valid -> Total: {avg_val_loss:.6f} | Recon: {avg_val_recon:.6f} | KL: {avg_val_kl:.4f}")
        # Note: kl_weight is now part of the model's loss config, not an arg used every step
        print(f"  (KL Weight in model config: {args.kl_weight:.2e})")

        # --- Early Stopping Logic (Remains the same) ---
        if avg_val_loss < best_val_loss - args.early_stopping_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            output_path = Path(args.ldm_vae_weights_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"✅ Validation loss improved. Saving best model to: {output_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")

        if patience_counter >= args.early_stopping_patience:
            print("🛑 Early stopping triggered. Training finished.")
            break
    
    # --- 4. Final Visualization (Updated) ---
    print("Loading best model for final visualization...")
    if Path(args.ldm_vae_weights_path).exists():
        model.load_state_dict(torch.load(args.ldm_vae_weights_path, map_location=device, weights_only=False))
        model.eval()
        with torch.no_grad():
            original_sample = next(iter(dev_loader))[0].unsqueeze(0).to(device)
            
            # 4a. Get reconstruction from the model's forward pass
            reconstructed_sample, _ = model(original_sample)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(original_sample[0, 0].cpu().numpy(), cmap='viridis')
            axs[0].set_title("Original (u-velocity)")
            axs[1].imshow(reconstructed_sample[0, 0].cpu().numpy(), cmap='viridis')
            axs[1].set_title("Reconstruction (u-velocity)")
            
            image_save_path = Path(args.ldm_vae_weights_path).parent / "vae_reconstruction_quality.png"
            plt.savefig(image_save_path)
            print(f"✅ Reconstruction plot saved to: {image_save_path}")
    else:
        print("Could not find best model weights to create visualization.")

    loss_plot_path = Path(args.ldm_vae_weights_path).parent / "vae_loss_history.png"
    plot_loss_history(history, loss_plot_path)


if __name__ == "__main__":
    main()