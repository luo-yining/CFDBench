from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from tqdm.auto import tqdm

class VaeDataset(Dataset):
    """A wrapper dataset that extracts and resizes individual frames for VAE training."""
    def __init__(self, cfd_auto_dataset):
        self.frames = []
        
        # --- THE FIX: Define a transform to resize the images ---
        self.transform = T.Compose([
            T.Resize((64, 64), antialias=True) 
        ])
        # ---------------------------------------------------------

        print("Preprocessing and resizing frames for VAE training...")
        for i in tqdm(range(len(cfd_auto_dataset))):
            _, label, _ = cfd_auto_dataset[i]
            frame = label[:2, :, :] # We only need the u and v channels
            
            # Apply the resize transform to each frame
            resized_frame = self.transform(frame)
            self.frames.append(resized_frame)
            
        self.frames = torch.stack(self.frames)
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]