import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from typing import Tuple, List, Optional, Dict, Any

# Imports from the CFDBench project
from models.cfd_vae import CfdVae3
from dataset import get_auto_dataset
from train_vae import VaeDataset # Reuse the VaeDataset for consistent preprocessing
from args import Args


def visualize_cfd_latent_space(model, dataloader, num_samples=300):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    latents = []
    batch_indices = []  # To color-code by time/batch if usefuly
    
    print("Extracting latent representations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(latents) * batch[0].size(0) >= num_samples:
                break
                
            data = batch[0].unsqueeze(0).to(device)
            # Encode to latent space
            posterior = model.vae.encode(data)
            latent_mean = posterior.latent_dist.mean  # Shape: [batch, 4, H, W]
            
            # Flatten spatial dimensions: [batch, 4, H, W] -> [batch, 4*H*W]
            latent_flat = latent_mean.view(latent_mean.size(0), -1)
            latents.append(latent_flat.cpu().numpy())
            
            # Keep track of which batch each sample came from
            batch_indices.extend([batch_idx] * data.size(0))
    
    # Combine all latents
    all_latents = np.concatenate(latents, axis=0)[:num_samples]
    batch_indices = np.array(batch_indices)[:num_samples]
    
    print(f"Latent space shape: {all_latents.shape}")
    print(f"Running t-SNE (this may take a minute)...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                n_iter=1000, learning_rate=200)
    latents_2d = tsne.fit_transform(all_latents)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Color by sample index (temporal progression)
    scatter1 = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                              c=range(len(latents_2d)), cmap='viridis', 
                              alpha=0.7, s=20)
    axes[0].set_title('CFD Latent Space (t-SNE)\nColored by Sample Order')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Sample Index')
    
    # Plot 2: Color by batch (time steps)
    scatter2 = axes[1].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                              c=batch_indices, cmap='plasma', 
                              alpha=0.7, s=20)
    axes[1].set_title('CFD Latent Space (t-SNE)\nColored by Time Step')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Batch Index')
    
    plt.tight_layout()
    plt.show()
    
    return latents_2d, all_latents


def analyze_latent_distribution(model, dataloader, device, num_samples=200):
    """
    Analyzes and visualizes the latent space distribution of a VAE.
    """
    model.eval()
    latents = []
    
    with torch.no_grad():
        sample_count = 0
        # The dataloader already returns batches of data
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # The batch is already in the correct format (e.g., [16, 2, 64, 64])
            data = batch.to(device)
            
            # Encode the entire batch at once
            posterior = model.vae.encode(data)
            latent = posterior.latent_dist.mean
            latents.append(latent.cpu())
            
            sample_count += data.shape[0] # Increment by the batch size
    
    # Combine the latents from all batches
    all_latents = torch.cat(latents, dim=0)
    
    # Ensure we have exactly the desired number of samples
    all_latents = all_latents[:num_samples]
    
    print(f"Latent space shape: {all_latents.shape}")
    print(f"Latent mean: {all_latents.mean().item():.4f}")
    print(f"Latent std: {all_latents.std().item():.4f}")
    
    # --- Plotting Code ---
    
    # Plot comprehensive analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comprehensive Latent Space Distribution Analysis", fontsize=16)
    
    # Overall distribution
    latent_flat = all_latents.reshape(-1).numpy()
    # MODIFICATION 1: Increased bins and set x-axis range
    axes[0, 0].hist(latent_flat, bins=200, alpha=0.7, edgecolor='black', density=True)
    axes[0, 0].set_xlim(-2, 2)
    axes[0, 0].set_title('Overall Latent Distribution')
    axes[0, 0].set_xlabel('Latent Value')
    axes[0, 0].set_ylabel('Density (Relative Frequency)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-channel analysis
    num_channels = all_latents.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_channels))
    for c in range(num_channels):
        channel_data = all_latents[:, c].reshape(-1).numpy()
        # MODIFICATION 2: Increased bins
        axes[0, 1].hist(channel_data, bins=100, alpha=0.6, 
                        label=f'Channel {c}', color=colors[c], edgecolor='black', density=True)
    
    # MODIFICATION 2: Set x-axis range for per-channel plot
    axes[0, 1].set_xlim(-2, 2)
    axes[0, 1].set_title('Per-Channel Distributions')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Latent Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spatial average per sample
    spatial_avg = all_latents.mean(dim=[2, 3]) # Average over spatial dimensions
    box_data = [spatial_avg[:, c].numpy() for c in range(spatial_avg.shape[1])]
    axes[0, 2].boxplot(box_data, labels=[f'Ch{c}' for c in range(len(box_data))])
    axes[0, 2].set_title('Spatial Average per Channel')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Spatial Average')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Channel correlations
    if spatial_avg.shape[1] > 1:
        channel_corr = torch.corrcoef(spatial_avg.T)
        im = axes[1, 0].imshow(channel_corr.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        
        # MODIFICATION 3: Set integer ticks for the correlation matrix
        tick_locs = np.arange(num_channels)
        axes[1, 0].set_xticks(tick_locs)
        axes[1, 0].set_yticks(tick_locs)
        axes[1, 0].set_xticklabels(tick_locs)
        axes[1, 0].set_yticklabels(tick_locs)
        
        axes[1, 0].set_title('Channel Correlations')
        axes[1, 0].set_xlabel('Channel')
        axes[1, 0].set_ylabel('Channel')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Latent evolution over time (assuming samples are a time series)
    time_evolution = spatial_avg.mean(dim=1) # Average across all channels
    axes[1, 1].plot(time_evolution.numpy(), 'b-', alpha=0.7, linewidth=2)
    axes[1, 1].set_title('Latent Evolution Over Time')
    axes[1, 1].set_xlabel('Sample Index (Time)')
    axes[1, 1].set_ylabel('Mean Latent Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Standard deviation per channel over time
    time_std = spatial_avg.std(dim=1)
    axes[1, 2].plot(time_std.numpy(), 'r-', alpha=0.7, linewidth=2)
    axes[1, 2].set_title('Latent Variability Over Time')
    axes[1, 2].set_xlabel('Sample Index (Time)')
    axes[1, 2].set_ylabel('Latent Std Dev')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def latent_interpolation(model, sample1_raw, sample2_raw, num_steps=8):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Add batch dimension
    sample1 = sample1_raw.unsqueeze(0).to(device)  # [1, 2, 64, 64]
    sample2 = sample2_raw.unsqueeze(0).to(device)  # [1, 2, 64, 64]
    
    with torch.no_grad():
        # Encode both samples
        posterior1 = model.vae.encode(sample1)
        posterior2 = model.vae.encode(sample2)
        
        latent1 = posterior1.latent_dist.mean
        latent2 = posterior2.latent_dist.mean
        
        print(f"Latent shape: {latent1.shape}")
        
        # Create interpolation
        alphas = np.linspace(0, 1, num_steps)
        
        fig, axes = plt.subplots(3, num_steps, figsize=(20, 8))
        
        for i, alpha in enumerate(alphas):
            # Interpolate in latent space
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
            
            # Decode back to image space
            reconstructed = model.vae.decode(interpolated_latent).sample
            
            # Plot u-velocity
            axes[0, i].imshow(reconstructed[0, 0].cpu().numpy(), cmap='viridis')
            axes[0, i].set_title(f'U-vel α={alpha:.2f}')
            axes[0, i].axis('off')
            
            # Plot v-velocity
            axes[1, i].imshow(reconstructed[0, 1].cpu().numpy(), cmap='viridis')
            axes[1, i].set_title(f'V-vel α={alpha:.2f}')
            axes[1, i].axis('off')
            
            # Plot velocity magnitude
            u_vel = reconstructed[0, 0].cpu().numpy()
            v_vel = reconstructed[0, 1].cpu().numpy()
            magnitude = np.sqrt(u_vel**2 + v_vel**2)
            axes[2, i].imshow(magnitude, cmap='plasma')
            axes[2, i].set_title(f'Mag α={alpha:.2f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()




class VAELatentAssessment:
    """
    A comprehensive toolkit for assessing the latent space quality of our CfdVae,
    which is based on the Hugging Face diffusers.AutoencoderKL model.
    """
    
    def __init__(self, vae_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = vae_model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def encode_batch(self, x: torch.Tensor):
        """Encodes a batch and returns the posterior distribution."""
        with torch.no_grad():
            x = x.to(self.device)
            # The diffusers VAE returns the distribution object directly
            return self.model.vae.encode(x).latent_dist

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes."""
        with torch.no_grad():
            z = z.to(self.device)
            # The diffusers VAE returns a dictionary-like object with a .sample attribute
            return self.model.vae.decode(z).sample

    def plot_latent_traversals(self, x: torch.Tensor, n_steps: int = 7, 
                                 range_scale: float = 2.0, save_path: Path = None):
        """
        Visualize how traversing each latent CHANNEL affects the reconstruction.
        This is adapted for a spatial latent space.
        """
        posterior = self.encode_batch(x.unsqueeze(0))
        z_base = posterior.mean
        num_channels = z_base.shape[1]
        
        fig, axes = plt.subplots(num_channels, n_steps, figsize=(n_steps * 2, num_channels * 2))
        fig.suptitle("Latent Space Traversal per Channel", fontsize=16)

        for dim in range(num_channels):
            traversal_vals = torch.linspace(-range_scale, range_scale, n_steps, device=self.device)
            
            for step, val in enumerate(traversal_vals):
                z_traversal = z_base.clone()
                # Modify the mean activation of the entire channel
                z_traversal[:, dim, :, :] += val
                
                x_recon = self.decode_batch(z_traversal)
                # We plot the u-velocity (channel 0) of the reconstruction
                img = x_recon[0, 0].cpu().numpy()
                
                ax = axes[dim, step]
                ax.imshow(img, cmap='viridis')
                ax.axis('off')
                
                if step == n_steps // 2:
                    ax.set_title("Original (Mean)", fontsize=10)
                else:
                    ax.set_title(f"Val: {val:.1f}", fontsize=10)

                if step == 0:
                    ax.text(-0.1, 0.5, f'Channel {dim}', transform=ax.transAxes, 
                            ha='right', va='center', fontsize=12, rotation=90)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_reconstruction_comparison(self, dataloader, num_pairs: int = 5, save_path: Path = None):
        """Plots a side-by-side comparison of original and reconstructed images."""
        originals, reconstructions = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_pairs:
                    break
                sample = batch[0].unsqueeze(0).to(self.device)
                reconstruction = self.model(sample).sample
                originals.append(sample[0].cpu())
                reconstructions.append(reconstruction[0].cpu())

        fig, axes = plt.subplots(num_pairs, 2, figsize=(6, num_pairs * 3))
        fig.suptitle("Original vs. Reconstruction", fontsize=16)
        for i in range(num_pairs):
            axes[i, 0].imshow(originals[i][0].numpy(), cmap='viridis')
            axes[i, 0].set_title(f"Original #{i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(reconstructions[i][0].numpy(), cmap='viridis')
            axes[i, 1].set_title(f"Reconstruction #{i+1}")
            axes[i, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_latent_channels(self, x: torch.Tensor, save_path: Path = None):
        """
        Encodes a single sample and visualizes each channel of its latent representation.
        """
        posterior = self.encode_batch(x.unsqueeze(0))
        # Get the mean of the latent distribution, shape [1, C, H, W]
        z_mean = posterior.mean
        
        num_channels = z_mean.shape[1]
        
        fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 4, 4))
        if num_channels == 1: # Ensure axes is always a list for consistent indexing
            axes = [axes]
            
        fig.suptitle("Latent Space Channel Activations for a Single Sample", fontsize=16)

        # Get global min and max across all channels for consistent color scaling
        vmin = z_mean.min()
        vmax = z_mean.max()

        for dim in range(num_channels):
            # Get the spatial map for the current channel
            latent_map = z_mean[0, dim].cpu().numpy()
            
            ax = axes[dim]
            im = ax.imshow(latent_map, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Channel {dim}')
            ax.axis('off')

        # --- CORRECTED PART ---
        # Adjust the main plot area to make space on the right for the colorbar
        fig.subplots_adjust(right=0.85)
        
        # Create a new, dedicated axis for the colorbar
        # The arguments are [left, bottom, width, height] in figure-relative coordinates
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        
        # Add the colorbar to this new axis
        fig.colorbar(im, cax=cbar_ax)
        # --------------------

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


    def assess_posterior_collapse(self, dataloader, num_samples: int = 500, threshold: float = 0.01) -> dict:
        """
        Assess posterior collapse by analyzing the KL divergence.
        For a spatial VAE, we average the KL divergence across the spatial dimensions.
        """
        kl_divs = []
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                posterior = self.encode_batch(batch)
                # .kl() returns KL per sample, shape [B, 4, 8, 8]
                # We average over the spatial dimensions and then the batch
                kl_per_sample = posterior.kl()
                kl_divs.append(kl_per_sample.cpu())
                sample_count += batch.shape[0]

        all_kls = torch.cat(kl_divs)
        mean_kl = all_kls.mean().item()
        
        is_collapsed = mean_kl < threshold
        
        print(f"Average KL Divergence per sample: {mean_kl:.6f}")
        if is_collapsed:
            print(f"🚨 WARNING: Posterior collapse may be occurring (Mean KL < {threshold}).")
        else:
            print("✅ Latent space appears to be active.")

        return {"mean_kl_divergence": mean_kl, "is_collapsed": is_collapsed}

def main():
    """Example usage of the VAELatentAssessment toolkit."""
    args = Args().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Load Model and Data ---
    print("Loading trained VAE and test data...")
    model = CfdVae3(in_chan=2, out_chan=2, latent_dim=args.ldm_latent_dim)
    model.load_state_dict(torch.load(args.ldm_vae_weights_path, map_location=device))

    _, _, test_data_raw = get_auto_dataset(
        data_dir=Path(args.data_dir), data_name=args.data_name, delta_time=args.delta_time,
        norm_props=True, norm_bc=True, load_splits=['test']
    )
    test_dataset = VaeDataset(test_data_raw)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    # --- 2. Initialize and Use the Assessment Toolkit ---
    assessor = VAELatentAssessment(model, device=device)

    # Run a reconstruction quality check
    print("\n--- Running Reconstruction Test ---")
    assessor.plot_reconstruction_comparison(test_loader, num_pairs=5)

    # Run a latent traversal
    print("\n--- Running Latent Traversal Test ---")
    sample_for_traversal, *_ = test_dataset[len(test_dataset) // 2]
    assessor.plot_latent_traversals(sample_for_traversal)
    
    # Run a posterior collapse check
    print("\n--- Running Posterior Collapse Test ---")
    assessor.assess_posterior_collapse(test_loader)

if __name__ == "__main__":
    main()


