import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import torchvision.transforms as T
import json


# Imports from the CFDBench project
from args import Args # Import the main Args class
from dataset import get_auto_dataset
from dataset.vae import VaeDataset



def compute_dataset_mean_std(dataloader: DataLoader) -> (torch.Tensor, torch.Tensor):
    """
    Calculates the mean and standard deviation of a dataset per channel.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    print("Calculating dataset statistics for normalization...")
    for data in tqdm(dataloader, desc="Calculating Stats"):
        # Sums are calculated channel-wise, across all other dimensions
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.clamp((channels_squared_sum / num_batches - mean**2)**0.5, min=1e-8)
    
    return mean, std

if __name__ == '__main__':

    args = Args().parse_args()

    train_data_raw, dev_data_raw, _ = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        load_splits=['train']
    )
    assert train_data_raw is not None
    
    # --- MODIFIED PART: Compute or Load Normalization Stats ---
    stats_path = Path(args.output_dir) / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_path.exists() and not args.clear_cache:
        print(f"Loading normalization stats from {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            mean = torch.tensor(stats['mean'])
            std = torch.tensor(stats['std'])
    else:
        # Create a temporary dataset and loader to calculate stats
        temp_dataset = VaeDataset(train_data_raw, mean=torch.tensor([0.0, 0.0]), std=torch.tensor([1.0, 1.0]))
        temp_loader = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)
        mean, std = compute_dataset_mean_std(temp_loader)
        
        # Save the stats for future runs
        with open(stats_path, 'w') as f:
            json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
        print(f"Saved normalization stats to {stats_path}")