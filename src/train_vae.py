import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tap import Tap
from tqdm import tqdm
import matplotlib.pyplot as plt

# Imports from the CFDBench project
from models.cfd_vae import CfdVae
from dataset import get_auto_dataset
from args import VaeArgs
from dataset.vae import VaeDataset



def main():
    """Main function to train the VAE."""
    args = VaeArgs().parse_args()
    print("--- Training VAE ---")
    print(args)

    # --- 1. Load Data ---
    print("Loading data...")
    train_data_raw, _, _ = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['train']
    )
    assert train_data_raw is not None
    
    train_dataset = VaeDataset(train_data_raw)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset created with {len(train_dataset)} frames of size {train_dataset[0].shape}.")

    # --- 2. Initialize Model and Optimizer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfdVae(in_chan=2, out_chan=2, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # --- 3. Training Loop ---
    print("Starting training loop...")
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0.0
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            
            # 1. Encode the input to get the latent distribution
            posterior = model.vae.encode(batch).latent_dist
            
            # 2. Sample from the latent distribution
            z = posterior.sample()
            
            # 3. Decode the latent sample to get the reconstruction
            reconstruction = model.vae.decode(z).sample
            # ---------------------------------------------------
            
            recon_loss = F.mse_loss(reconstruction, batch)
            kl_loss = posterior.kl().mean()
            
            loss = recon_loss + args.kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}", recon_loss=f"{recon_loss.item():.6f}", kl_loss=f"{kl_loss.item():.4f}")
            
        print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.6f}")

    # --- 4. Save the trained model ---
    output_path = Path(args.output_weights_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ VAE model weights saved to: {output_path}")

    # --- 5. Visualize a sample reconstruction ---
    print("Visualizing a sample reconstruction...")
    model.eval()
    with torch.no_grad():
        original_sample = next(iter(train_loader))[0].unsqueeze(0).to(device)
        # For visualization, we can use the full forward pass
        reconstructed_sample = model(original_sample).sample

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_sample[0, 0].cpu().numpy(), cmap='viridis')
        axs[0].set_title("Original (u-velocity)")
        axs[1].imshow(reconstructed_sample[0, 0].cpu().numpy(), cmap='viridis')
        axs[1].set_title("Reconstruction (u-velocity)")
        plt.show()

if __name__ == "__main__":
    main()
