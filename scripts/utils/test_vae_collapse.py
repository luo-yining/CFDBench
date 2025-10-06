import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# Imports from the CFDBench project
from models.cfd_vae import CfdVae2
from dataset import get_auto_dataset
from train_vae import VaeDataset # Reuse the VaeDataset for consistent preprocessing
from args import Args

def main():
    """Main function to test for VAE posterior collapse."""
    args = Args().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running VAE collapse test on device: {device}")

    # --- 1. Load the Trained VAE Model ---
    print(f"Loading VAE weights from: {args.ldm_vae_weights_path}")
    model = CfdVae2(in_chan=2, out_chan=2, latent_dim=args.ldm_latent_dim)
    state_dict = torch.load(args.ldm_vae_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 2. Load the Test Dataset ---
    print("Loading test data...")
    _, _, test_data_raw = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        load_splits=['test']
    )
    assert test_data_raw is not None
    test_dataset = VaeDataset(test_data_raw)

    # --- 3. Perform the Test on a Sample ---
    print("Performing reconstruction tests...")
    # Get a sample from the middle of the test set for a good representation
    sample_idx = len(test_dataset) // 2
    original_sample = test_dataset[sample_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        # Test 1: Standard Reconstruction (Encode -> Decode)
        posterior = model.vae.encode(original_sample).latent_dist
        latent_code = posterior.sample()
        standard_reconstruction = model.vae.decode(latent_code).sample

        # Test 2: Zero-Latent Reconstruction (Zeros -> Decode)
        # Create a zero tensor with the correct latent shape
        zero_latent = torch.zeros_like(latent_code)
        zero_latent_reconstruction = model.vae.decode(zero_latent).sample

    # --- 4. Visualize the Results ---
    print("Plotting results for comparison...")
    original_img = original_sample[0, 0].cpu().numpy()
    standard_recon_img = standard_reconstruction[0, 0].cpu().numpy()
    zero_recon_img = zero_latent_reconstruction[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("VAE Posterior Collapse Diagnostic", fontsize=16)

    # Plot Original
    axes[0].imshow(original_img, cmap='viridis')
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')

    # Plot Standard Reconstruction
    axes[1].imshow(standard_recon_img, cmap='viridis')
    axes[1].set_title("2. Standard Reconstruction")
    axes[1].axis('off')

    # Plot Zero-Latent Reconstruction
    axes[2].imshow(zero_recon_img, cmap='viridis')
    axes[2].set_title("3. Zero-Latent Reconstruction")
    axes[2].axis('off')
    
    # Calculate and display the difference for quantitative analysis
    diff = torch.nn.functional.mse_loss(standard_reconstruction, zero_latent_reconstruction)
    plt.figtext(0.5, 0.05, f"MSE between reconstructions: {diff.item():.6f}", ha="center", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Or use plt.savefig() on a server

if __name__ == "__main__":
    main()


### How to Use and Interpret the Results

    # In your src/ directory
    #python test_vae_collapse.py --data_name cylinder_prop_bc_geo
    
