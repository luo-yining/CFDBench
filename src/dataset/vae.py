import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class VaeDataset(Dataset):
    """
    A wrapper dataset that extracts and resizes frames for VAE training.
    Normalization is now optional and controlled by the mean and std parameters.
    """
    def __init__(self, cfd_auto_dataset, normalize=True):
        """
        Args:
            cfd_auto_dataset: The raw autoregressive dataset from CFDBench.
            normalize: bool. whether to normalize the data using mean, std previously calculated
        """
        self.raw_dataset = cfd_auto_dataset

        
        # Start with the mandatory resize transformation
        transforms_list = [
            T.Resize((64, 64), antialias=True)
        ]
        
        # Only add the normalization step if both mean and std are provided
        if normalize:
            # normalization based on dataset mean and std
            transforms_list.append(T.Normalize(mean=torch.tensor([1.891, 1.806]), std=torch.tensor([1.550, 1.574])))
            print("VaeDataset initialized WITH custom normalization.")
        else:
            print("VaeDataset initialized WITHOUT normalization (only resizing).")
            
        self.transform = T.Compose(transforms_list)
        # ---------------------------------------------

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        # The logic here remains the same, as it just applies the transform pipeline
        _, label, _ = self.raw_dataset[idx]
        frame = label[:2, :, :]
        
        # The transform will either be [Resize] or [Resize, Normalize]
        transformed_frame = self.transform(frame)
        
        return transformed_frame
