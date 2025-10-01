import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json

# Import for Automatic Mixed Precision (AMP)
from torch.amp import autocast
from torch.amp import GradScaler 
# -----------------------------------------------------------

# Imports from the CFDBench project
from models.latent_diffusion import LatentDiffusionCfdModel
from dataset import get_auto_dataset
from utils_auto import init_model 
from utils import plot_predictions, dump_json, load_best_ckpt
from args import Args

def evaluate_ldm(model: LatentDiffusionCfdModel, dataloader, device, output_dir: Path, plot_interval: int = 20):
    """
    Custom evaluation function for the Latent Diffusion Model.
    The mask is used here to ensure the loss is only calculated in valid fluid regions.
    """
    model.eval()
    total_nmse = 0.0
    
    print("=== Evaluating LDM ===")
    progress_bar = tqdm(dataloader, desc="Evaluation")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # The collate_fn now provides the mask as a separate item
            inputs, label, case_params, mask = batch
            inputs, label, case_params, mask = inputs.to(device), label.to(device), case_params.to(device), mask.to(device)
            
            # The generate function only receives the 2-channel velocity input
            generated_frame = model.generate(inputs=inputs, case_params=case_params)
            
            # Apply the mask to both the prediction and the label for a fair comparison
            loss = F.mse_loss(generated_frame * mask, label * mask)
            nmse = loss / (torch.square(label * mask).mean() + 1e-8)
            total_nmse += nmse.item()
            
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

    avg_nmse = total_nmse / len(dataloader)
    print(f"Evaluation NMSE: {avg_nmse:.6f}")
    return {"mean": {"nmse": avg_nmse}}

def train_ldm(model: LatentDiffusionCfdModel, train_loader, dev_loader, args, device):
    """ Main training loop for the Latent Diffusion Model. """
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize the Gradient Scaler for Automatic Mixed Precision
    scaler = GradScaler(device=device, enabled=args.use_mixed_precision)
    
    print("====== Training LDM ======")
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Mask is available from collate_fn but not used in the training forward pass
            inputs, label, case_params, _ = batch
            inputs, label, case_params = inputs.to(device), label.to(device), case_params.to(device)
            
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing
            
            # We specify the device type and target dtype explicitly.
            with autocast(device_type=device, dtype=torch.float16, enabled=args.use_mixed_precision):
                outputs = model(inputs=inputs, label=label, case_params=case_params)
                loss = outputs["loss"]["mse"]
            # --------------------------------------------
            
            # Use the scaler for the backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        # --- Evaluation ---
        if (epoch + 1) % args.eval_interval == 0:
            output_dir = Path(args.output_dir) / "ldm_training"
            ckpt_dir = output_dir / f"ckpt-{epoch}"
            
            dev_scores = evaluate_ldm(model, dev_loader, device, ckpt_dir)
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            
            # Save the U-Net checkpoint
            model.unet.save_pretrained(ckpt_dir / "unet")
            print(f"Saved U-Net checkpoint to {ckpt_dir / 'unet'}")

def main():
    """Main function to train the Latent Diffusion Model."""
    args = Args().parse_args()
    args.model = "latent_diffusion"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Data ---
    print("Loading raw dataset for LDM training...")
    train_data_raw, dev_data_raw, _ = get_auto_dataset(
        data_dir=Path(args.data_dir), data_name=args.data_name, delta_time=args.delta_time,
        norm_props=True, norm_bc=True, load_splits=['train', 'dev']
    )
    
    # --- 2. Initialize the LDM ---
    print(f"Using hardcoded scaling factor from args: {args.ldm_scaling_factor}")
    args.in_chan = 2 # Conditioning on u, v only
    model = init_model(args)
    model.to(device)

    # --- 3. Create Final DataLoaders ---
    def collate_fn_ldm(batch):
        inputs_raw, labels_raw, case_params_dict = zip(*batch)
        
        inputs = torch.stack(inputs_raw)
        labels = torch.stack(labels_raw)
        
        # Separate the mask from the input velocities
        input_velocities = inputs[:, :2]  # Channels 0, 1 are u, v
        mask = inputs[:, 2:]             # Channel 2 is the mask
        
        # The label should only be the velocities
        output_velocities = labels[:, :2]
        
        keys = case_params_dict[0].keys()
        case_params = torch.tensor([[d[k] for k in keys] for d in case_params_dict])
        
        # Return the mask as a separate item
        return input_velocities, output_velocities, case_params, mask

    train_loader = DataLoader(train_data_raw, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ldm, num_workers=4)
    dev_loader = DataLoader(dev_data_raw, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_ldm, num_workers=4)

    # --- 4. Start Training ---
    train_ldm(model, train_loader, dev_loader, args, device)

if __name__ == "__main__":
    main()

