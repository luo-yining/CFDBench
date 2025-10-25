import torch
import numpy as np
import matplotlib.pyplot as plt

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
    axes[0, 0].set_xlim(-1, 1)
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
    axes[0, 1].set_xlim(-1, 1)
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


if __name__ == '__main__':
    pass