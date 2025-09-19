import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tap import Tap
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Imports from the CFDBench project
from models.latent_diffusion import LatentDiffusionCfdModel
from dataset import get_auto_dataset
from utils_auto import init_model # We can reuse the model initializer
from utils import plot_predictions, dump_json, load_best_ckpt

def evaluate_ldm(model: LatentDiffusionCfdModel, dataloader, device, output_dir: Path, plot_interval: int = 20):
    """
    Custom evaluation function for the Latent Diffusion Model.
    This function performs the full generation process to get a predicted frame.
    """
    model.eval()
    total_nmse = 0.0
    
    print("=== Evaluating LDM ===")
    progress_bar = tqdm(dataloader, desc="Evaluation")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            inputs, label, case_params = batch
            inputs, label, case_params = inputs.to(device), label.to(device), case_params.to(device)
            
            # Perform the full denoising process to generate the next frame
            generated_frame = model.generate(inputs=inputs, case_params=case_params)
            
            # Now, compare the generated frame to the true next frame (the label)
            loss = F.mse_loss(generated_frame, label)
            nmse = loss / (torch.square(label).mean() + 1e-8)
            total_nmse += nmse.item()
            
            # Plot a sample for visual inspection
            if i % plot_interval == 0:
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
    """
    Main training loop for the Latent Diffusion Model.
    """
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr)
    
    print("====== Training LDM ======")
    for epoch in range(args.num_epochs):
        model.train() # Set the U-Net to training mode
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        total_loss = 0.0
        
        for batch in progress_bar:
            inputs, label, case_params = batch
            inputs, label, case_params = inputs.to(device), label.to(device), case_params.to(device)
            
            optimizer.zero_grad()
            
            # The forward pass calculates the noise prediction loss
            outputs = model(inputs=inputs, label=label, case_params=case_params)
            loss = outputs["loss"]["mse"] # Use the mse loss on the noise
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average training loss: {avg_loss:.6f}")
        
        # --- Evaluation ---
        if (epoch + 1) % args.eval_interval == 0:
            output_dir = Path(args.output_dir)
            ckpt_dir = output_dir / f"ckpt-{epoch}"
            
            dev_scores = evaluate_ldm(model, dev_loader, device, ckpt_dir)
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            
            # Save U-Net checkpoint
            model.unet.save_pretrained(ckpt_dir / "unet")
            print(f"Saved U-Net checkpoint to {ckpt_dir / 'unet'}")

def main():
    # We can reuse the Args class after adding our LDM parameters
    from args import Args
    args = Args().parse_args()
    args.model = "latent_diffusion" # Hardcode for this script
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Data Loading (reusing existing functions) ---
    train_data_raw, dev_data_raw, _ = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=True, norm_bc=True,
        load_splits=['train', 'dev']
    )
    # The LDM needs a different collate_fn that doesn't strip the mask
    def collate_fn_ldm(batch):
        inputs, labels, case_params_dict = zip(*batch)
        inputs = torch.stack(inputs)[:, :2] # Use only u, v for conditioning
        labels = torch.stack(labels)[:, :2] # Use only u, v for target
        
        keys = case_params_dict[0].keys()
        case_params = torch.tensor([[d[k] for k in keys] for d in case_params_dict])
        return inputs, labels, case_params

    train_loader = DataLoader(train_data_raw, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_ldm)
    dev_loader = DataLoader(dev_data_raw, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_ldm)

    # --- Model Initialization (reusing existing function) ---
    model = init_model(args).to(device)

    # --- Start Training ---
    train_ldm(model, train_loader, dev_loader, args, device)

if __name__ == "__main__":
    main()



    # Run the new training script
    # python train_ldm.py --data_name cylinder_prop --num_epochs 200 --lr 1e-4
    
