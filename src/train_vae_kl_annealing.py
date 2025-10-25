import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import shutil
import numpy as np

# Imports from the CFDBench project
from models.cfd_vae import CfdVae2
from dataset import get_auto_dataset
from args import Args # Import the main Args class
from dataset.vae import VaeDataset

def main():
    """Main function to train the VAE."""
    args = Args().parse_args()
    print("--- Training VAE with KL Annealing ---")
    print(args)

    # --- 1. Load Data ---
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
    
    train_dataset = VaeDataset(train_data_raw)
    dev_dataset = VaeDataset(dev_data_raw)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train dataset created with {len(train_dataset)} frames.")
    print(f"Validation dataset created with {len(dev_dataset)} frames.")

    # --- 2. Initialize Model and Optimizer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfdVae2(in_chan=2, out_chan=2, latent_dim=args.ldm_latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # --- 3. Training Loop with KL Annealing & Early Stopping ---
    print("Starting training loop...")
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        # --- KL ANNEALING LOGIC ---
        # Linearly increase the KL weight from 0 to its full value over N epochs
        current_kl_weight = args.vae_kl_weight * min(1.0, epoch / args.vae_kl_annealing_epochs)
        progress_bar.set_postfix_str(f"KL Weight: {current_kl_weight:.2e}")
        # -------------------------
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            posterior = model.vae.encode(batch).latent_dist
            z = posterior.sample()
            reconstruction = model.vae.decode(z).sample
            
            recon_loss = F.mse_loss(reconstruction, batch)
            kl_loss = posterior.kl().mean()
            # Use the dynamically calculated KL weight
            loss = recon_loss + current_kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                posterior = model.vae.encode(batch).latent_dist
                z = posterior.sample()
                reconstruction = model.vae.decode(z).sample
                recon_loss = F.mse_loss(reconstruction, batch)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + current_kl_weight * kl_loss # Use same weight for consistency
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(dev_loader)
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss - args.early_stopping_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            output_path = Path(args.ldm_vae_weights_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"✅ Val loss improved. Saving best model to: {output_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print("🛑 Early stopping triggered.")
            break
    
    # --- Final Visualization ---
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
if __name__ == "__main__":
    main()