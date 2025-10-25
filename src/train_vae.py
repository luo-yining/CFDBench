import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tap import Tap
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import numpy as np

# Imports from the CFDBench project
from models.cfd_vae import CfdVae2, CfdVae3, CfdVaeLite
from dataset import get_auto_dataset
from args import Args
from dataset.vae import VaeDataset


def plot_loss_history(history, save_path):
    """
    Plots the training (per-step) and validation (per-epoch) loss curves.
    A moving average is used to smooth the noisy training loss for better visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("VAE Training Loss History (per Step)", fontsize=16)

    # Helper function for moving average
    def moving_average(data, window_size=100):
        if len(data) < window_size:
            return np.array([]) # Not enough data for a moving average
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    # --- Plot Total Loss ---
    train_steps = np.arange(len(history['train_total']))
    axes[0].plot(train_steps, history['train_total'], label='Train Loss (Raw)', alpha=0.2, color='lightblue')
    
    # Plot smoothed training loss
    avg_train_loss = moving_average(history['train_total'])
    # Adjust steps for the moving average window to be centered
    if avg_train_loss.any():
        avg_steps = np.arange(len(avg_train_loss)) + (100 // 2)
        axes[0].plot(avg_steps, avg_train_loss, label='Train Loss (Smoothed)', color='blue')
    
    # Plot validation loss as a staircase plot, aligned with training steps
    axes[0].step(history['val_steps'], history['val_total'], label='Validation Loss', where='post', color='orange', linewidth=2)
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss (Log Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # --- Plot Reconstruction Loss ---
    axes[1].plot(train_steps, history['train_recon'], label='Train Recon Loss (Raw)', alpha=0.2, color='lightblue')
    avg_train_recon = moving_average(history['train_recon'])
    if avg_train_recon.any():
        avg_steps = np.arange(len(avg_train_recon)) + (100 // 2)
        axes[1].plot(avg_steps, avg_train_recon, label='Train Recon Loss (Smoothed)', color='blue')
    axes[1].step(history['val_steps'], history['val_recon'], label='Validation Recon Loss', where='post', color='orange', linewidth=2)
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Training Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # --- Plot KL Loss ---
    axes[2].plot(train_steps, history['train_kl'], label='Train KL Loss (Raw)', alpha=0.2, color='lightblue')
    avg_train_kl = moving_average(history['train_kl'])
    if avg_train_kl.any():
        avg_steps = np.arange(len(avg_train_kl)) + (100 // 2)
        axes[2].plot(avg_steps, avg_train_kl, label='Train KL Loss (Smoothed)', color='blue')
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
    args = Args().parse_args()
    print("--- Training VAE ---")
    print(args)

    # --- 1. Load Data ---
    print("Loading data...")
    # Load both train and dev (validation) splits for early stopping
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

      # --- history dictionary for per-step logging ---
    history = {
        'train_total': [], 'train_recon': [], 'train_kl': [],
        'val_total': [], 'val_recon': [], 'val_kl': [],
        'val_steps': [] # To align validation epochs with training steps
    }
    # -----------------------------------------------------------------

    # --- 2. Initialize Model and Optimizer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfdVaeLite(in_chan=2, out_chan=2, latent_dim=args.ldm_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.vae_weight_decay)

    # --- Initialize the Scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', # It will monitor the validation loss for a minimum
        factor=args.lr_scheduler_factor, # Factor to reduce LR by (e.g., 0.5)
        patience=args.lr_scheduler_patience, # How many epochs to wait (e.g., 5)
        verbose=True # Print a message to the console when the LR is changed
    )
    # -----------------------------------------------

      # --- 3. Training Loop with Detailed Logging ---
    print("Starting training loop...")
    best_val_loss = np.inf
    patience_counter = 0
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train() # Set model to training mode
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        total_train_loss, total_train_recon, total_train_kl = 0.0, 0.0, 0.0
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            posterior = model.vae.encode(batch).latent_dist
            z = posterior.sample()
            reconstruction = model.vae.decode(z).sample
            
            recon_loss = F.mse_loss(reconstruction, batch)
            kl_loss = posterior.kl().mean()
            # Use the fixed KL weight directly from args
            loss = recon_loss + args.vae_kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()

             # --- Log per-batch loss ---
            history['train_total'].append(loss.item())
            history['train_recon'].append(recon_loss.item())
            history['train_kl'].append(kl_loss.item())
            global_step += 1
            
            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            total_train_kl += kl_loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        # Calculate average losses for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon = total_train_recon / len(train_loader)
        avg_train_kl = total_train_kl / len(train_loader)

 
        # --- Validation Step with Detailed Logging ---
        model.eval()
        total_val_loss, total_val_recon, total_val_kl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                posterior = model.vae.encode(batch).latent_dist
                z = posterior.mean
                reconstruction = model.vae.decode(z).sample
                recon_loss = F.mse_loss(reconstruction, batch)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + args.vae_kl_weight * kl_loss
                
                total_val_loss += loss.item()
                total_val_recon += recon_loss.item()
                total_val_kl += kl_loss.item()
        
        avg_val_loss = total_val_loss / len(dev_loader)
        avg_val_recon = total_val_recon / len(dev_loader)
        avg_val_kl = total_val_kl / len(dev_loader)

        
        # --- Log validation score at the current step number ---
        history['val_total'].append(avg_val_loss)
        history['val_recon'].append(avg_val_recon)
        history['val_kl'].append(avg_val_kl)
        history['val_steps'].append(global_step)


        # --- Step the Scheduler ---
        # The scheduler's step is called with the validation loss after each epoch.
        # It will automatically reduce the LR if the loss plateaus.
        scheduler.step(avg_val_loss)
        
        # --- Print detailed log ---
        print(f"Epoch {epoch+1}:")
        print(f"  Train -> Total: {avg_train_loss:.6f} | Recon: {avg_train_recon:.6f} | KL: {avg_train_kl:.4f}")
        print(f"  Valid -> Total: {avg_val_loss:.6f} | Recon: {avg_val_recon:.6f} | KL: {avg_val_kl:.4f}")
        print(f"  (KL Weight: {args.vae_kl_weight:.2e})")
        # ----------------------------------------

        # --- Early Stopping Logic ---
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
    
    # --- 4. Final Visualization using the best model ---
    print("Loading best model for final visualization...")
    # Make sure the weights file exists before trying to load it
    if Path(args.ldm_vae_weights_path).exists():
        model.load_state_dict(torch.load(args.ldm_vae_weights_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # Use the dev_loader to get a consistent sample for visualization
            original_sample = next(iter(dev_loader))[0].unsqueeze(0).to(device)
            reconstructed_sample = model(original_sample).sample

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

    # --- Plot and save the loss history ---
    loss_plot_path = output_path.parent / "vae_loss_history.png"
    plot_loss_history(history, loss_plot_path)
    # ----------------------------------------------------


if __name__ == "__main__":
    main()
