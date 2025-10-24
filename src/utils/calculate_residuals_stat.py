import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
import gc

# --- PATH LOGIC FOR src/utils/ ---
# If this script is in src/utils/, we might not need sys.path manipulation
# if the top-level directory (containing src) is in the PYTHONPATH or
# if scripts are run from the top-level directory using e.g., python -m src.utils.calculate_residual_stats
# However, keeping a robust path setup is safer.
# Path(__file__).resolve().parent is 'src/utils'
# .parent is 'src'
# .parent.parent is the project root
project_root = Path(__file__).resolve().parent.parent.parent
# Add project root to sys.path to allow imports like 'from src.args import Args'
sys.path.insert(0, str(project_root))
# --- END PATH LOGIC ---

# CFDBench Imports - Use absolute imports from src now
from src.args import Args
from src.dataset import get_auto_dataset
try:
    from src.dataset.wrapper import GenCastWrapperDataset
except ImportError:
     print("ERROR: GenCastWrapperDataset not found.")
     print(f"Looked in sys.path: {sys.path}")
     print("Ensure src/dataset/wrapper.py exists.")
     sys.exit(1)


# Constants
RESIDUAL_STATS_FILENAME = "residual_stats.pt" # Default filename to save stats

def main():
    """Calculates and saves the mean and std dev of the residual (Xt - Xt-1)."""
    args = Args().parse_args()
    print("--- Calculating Residual Statistics (src/utils/) ---")
    print(f"Dataset: {args.data_name}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Delta Time (for t vs t-1): {args.delta_time}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load RAW Training Data ---
    print("Loading RAW training dataset...")
    train_data_raw, _, _ = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        load_splits=['train']
    )
    if not train_data_raw:
        print("Error: Could not load training data.")
        return
    print(f"Raw training dataset loaded with {len(train_data_raw)} (t-1, t) pairs.")

    # --- 2. Wrap Dataset ---
    wrapped_train_dataset = GenCastWrapperDataset(train_data_raw)
    if len(wrapped_train_dataset) == 0:
        print("Error: Wrapped dataset is empty.")
        return
    print(f"Wrapped dataset created with {len(wrapped_train_dataset)} valid (t-2, t-1, t) samples.")

    # --- 3. Create DataLoader ---
    loader = DataLoader(
        wrapped_train_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --- 4. Pass 1: Calculate Mean ---
    print("\n--- Pass 1: Calculating Mean Residual ---")
    sum_residuals = 0.0
    total_samples = 0
    num_channels = -1

    progress_bar_mean = tqdm(loader, desc="Calculating Mean")
    for batch in progress_bar_mean:
        inputs = batch['inputs'].to(device)
        label = batch['label'].to(device)
        inputs_vel = inputs[:, :args.out_chan, :, :]
        label_vel = label[:, :args.out_chan, :, :]

        if num_channels == -1:
            num_channels = inputs_vel.shape[1]
            sum_residuals = torch.zeros(num_channels, device=device)
            print(f"Detected {num_channels} velocity channels.")

        batch_residuals = label_vel - inputs_vel
        sum_residuals += batch_residuals.sum(dim=(0, 2, 3))
        total_samples += batch_residuals.numel() // num_channels

        del inputs, label, inputs_vel, label_vel, batch_residuals
        if device.type == 'cuda': torch.cuda.empty_cache()

    if total_samples == 0:
        print("Error: No samples processed. Cannot calculate mean.")
        return

    mean_residual = sum_residuals / total_samples
    print(f"Mean calculated across {total_samples * num_channels} data points ({total_samples} per channel).")
    print(f"Mean Residual per channel: {mean_residual.cpu().numpy()}")

    # --- 5. Pass 2: Calculate Standard Deviation ---
    print("\n--- Pass 2: Calculating Standard Deviation ---")
    sum_sq_diff = torch.zeros_like(mean_residual)
    mean_residual_broadcast = mean_residual.view(1, -1, 1, 1)

    progress_bar_std = tqdm(loader, desc="Calculating Std Dev")
    for batch in progress_bar_std:
        inputs = batch['inputs'].to(device)
        label = batch['label'].to(device)
        inputs_vel = inputs[:, :args.out_chan, :, :]
        label_vel = label[:, :args.out_chan, :, :]

        batch_residuals = label_vel - inputs_vel
        diff = batch_residuals - mean_residual_broadcast
        sum_sq_diff += (diff ** 2).sum(dim=(0, 2, 3))

        del inputs, label, inputs_vel, label_vel, batch_residuals, diff
        if device.type == 'cuda': torch.cuda.empty_cache()

    variance = sum_sq_diff / total_samples
    std_dev_residual = torch.sqrt(variance)
    std_dev_residual = torch.clamp(std_dev_residual, min=1e-6)

    print(f"Std Dev Residual per channel: {std_dev_residual.cpu().numpy()}")

    # --- 6. Save Statistics ---
    stats_path = Path(getattr(args, 'residual_stats_path', RESIDUAL_STATS_FILENAME))
    output_dir_base = Path(args.output_dir) if args.output_dir else Path('.')
    # Save stats relative to the project root or output_dir, not inside src/utils
    stats_path_final = output_dir_base / stats_path
    stats_path_final.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving statistics to: {stats_path_final}")
    try:
        sample_data = wrapped_train_dataset[0]
        # Shape is [C+1, H, W], get H and W
        h_dim = sample_data['inputs'].shape[1]
        w_dim = sample_data['inputs'].shape[2]
    except Exception:
        h_dim, w_dim = -1, -1

    torch.save({
        'residual_mean': mean_residual.cpu(),
        'residual_std': std_dev_residual.cpu(),
        'data_name': args.data_name,
        'num_samples_used': total_samples // (h_dim * w_dim) if h_dim > 0 else -1,
        'dimensions': (h_dim, w_dim)
    }, stats_path_final)
    print("Statistics saved successfully.")

    # --- Clean up ---
    del loader, wrapped_train_dataset, train_data_raw
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
