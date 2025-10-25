import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple

# Assuming your CFDBench dataset classes are importable
# from dataset.base import CfdAutoDataset # Or the specific class like CavityFlowAutoDataset

class GenCastWrapperDataset(Dataset):
    """
    Wraps a CFDBench CfdAutoDataset to return three consecutive frames
    (t-2, t-1, t) needed for the GenCast-style model.

    It assumes the base dataset provides (X_{t-1}, X_t, case_params).
    """
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: An instance of a CfdAutoDataset subclass
                          (e.g., CavityFlowAutoDataset).
        """
        self.base_dataset = base_dataset
        
        # We need access to the original sequential data if possible,
        # but the base dataset pre-pairs t-1 and t.
        # We can reconstruct t-2 by looking at the *previous* index
        # in the base_dataset, BUT we must be careful about case boundaries.

        # Pre-calculate valid indices to avoid crossing case boundaries.
        self.valid_indices = []
        print("Pre-calculating valid indices for GenCastWrapperDataset...")
        for i in range(len(self.base_dataset)):
            # Get case ID for current index (i corresponds to t-1)
            # and previous index (i-1 corresponds to t-2)
            current_case_id = self.base_dataset.case_ids[i]
            if i > 0:
                previous_case_id = self.base_dataset.case_ids[i-1]
                # Only valid if the previous sample is from the same case
                if current_case_id == previous_case_id:
                    self.valid_indices.append(i)
            # The very first sample (i=0) is never valid because it has no t-2
        print(f"Wrapper dataset contains {len(self.valid_indices)} valid (t-2, t-1, t) samples.")


    def __len__(self) -> int:
        # The length is the number of valid indices we found
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
        - 'inputs_prev': X_{t-2} tensor [C, H, W] (including mask channel)
        - 'inputs':      X_{t-1} tensor [C, H, W] (including mask channel)
        - 'label':       X_{t}   tensor [C, H, W] (including mask channel)
        - 'case_params': Dictionary of case parameters for this sample.
                         (Will be converted to tensor in collate_fn)
        """
        # 'index' here refers to the index within self.valid_indices
        # Get the corresponding index in the base_dataset
        base_dataset_index = self.valid_indices[index]

        # Get the (X_{t-1}, X_t, case_params) tuple for the *current* step
        inputs_t_minus_1, label_t, case_params_t = self.base_dataset[base_dataset_index]

        # Get the (X_{t-2}, X_{t-1}, case_params) tuple for the *previous* step
        # We know base_dataset_index > 0 and it's within the same case
        # because of how we constructed valid_indices.
        inputs_t_minus_2, _, _ = self.base_dataset[base_dataset_index - 1]
        
        # Note: inputs_t_minus_1 and the label from the previous step should be identical.
        # We assume case_params are consistent for consecutive steps within a case.

        return {
            'inputs_prev': inputs_t_minus_2,  # This is X_{t-2}
            'inputs':      inputs_t_minus_1,  # This is X_{t-1}
            'label':       label_t,           # This is X_{t}
            'case_params': case_params_t      # Pass the dict, collate handles tensor conversion
        }



### How to Integrate into `train_auto_v2.py`

'''

1.  **Save the Wrapper:** Save the code above as `src/dataset/wrapper.py` (or similar).
2.  **Import:** In `src/train_auto_v2.py`, add the import:
    ```python
    from dataset.wrapper import GenCastWrapperDataset
    ```
3.  **Wrap the Datasets:** Modify the section where datasets are loaded:

    ```python
    # --- Load datasets (Original code) ---
    print("Loading datasets...")
    train_data_raw, dev_data_raw, test_data_raw = get_auto_dataset(
        data_dir=Path(args.data_dir),
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
    )
    # ... (asserts remain the same) ...

    # --- Wrap the datasets ---
    print("Wrapping datasets for GenCast model...")
    train_data = GenCastWrapperDataset(train_data_raw)
    dev_data = GenCastWrapperDataset(dev_data_raw)
    if test_data_raw: # Handle optional test set
        test_data = GenCastWrapperDataset(test_data_raw)
    else:
        test_data = None

    print(f"  Wrapped Train: {len(train_data)} examples")
    print(f"  Wrapped Dev: {len(dev_data)} examples")
    if test_data:
        print(f"  Wrapped Test: {len(test_data)} examples\n")

    # --- Create dataloaders (Use wrapped datasets) ---
    train_loader = DataLoader(
        train_data, # Use wrapped dataset
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_gen_cast, # Use the NEW collate_fn below
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    # ... (similarly update dev_loader and test_loader) ...
    ```

4.  **Create a New Collate Function:** The wrapper returns a dictionary. We need a `collate_fn` to stack these dictionary items correctly into batches. Add this function to `train_auto_v2.py`:

    ```python
    def collate_fn_gen_cast(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for the GenCastWrapperDataset.
        Stacks tensors and converts case_params dict to tensor.
        """
        # Stack the tensors for inputs_prev, inputs, label
        inputs_prev = torch.stack([item['inputs_prev'] for item in batch])
        inputs      = torch.stack([item['inputs'] for item in batch])
        labels      = torch.stack([item['label'] for item in batch])

        # Separate masks (last channel) from velocity fields (first two channels)
        mask        = inputs[:, -1:, :, :] # Assumes mask is always the last channel
        inputs_prev = inputs_prev[:, :-1, :, :]
        inputs      = inputs[:, :-1, :, :]
        labels      = labels[:, :-1, :, :]

        # Convert case_params dict to tensor (like the original collate_fn)
        case_params_list = [item['case_params'] for item in batch]
        keys = [k for k in case_params_list[0].keys() if k not in ["rotated", "dx", "dy"]]
        case_params = torch.tensor([[cp[k] for k in keys] for cp in case_params_list])

        return {
            "inputs_prev": inputs_prev, # X_{t-2} velocities
            "inputs":      inputs,      # X_{t-1} velocities
            "label":       labels,      # X_{t} velocities
            "mask":        mask,        # Mask from X_{t-1}
            "case_params": case_params, # Case parameters tensor
        }
    ```
    *Remember to use `collate_fn=collate_fn_gen_cast` when creating your DataLoaders.*

5.  **Update Training Loop:** Your training loop in `train_auto_v2.py` should now correctly receive batches with `inputs_prev`. Make sure your model's forward call matches:

    ```python
    # Inside the training loop of train_auto_v2.py
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass (model expects 'inputs_prev', 'inputs', 'label', etc.)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=args.use_mixed_precision):
            # The keys in the batch dictionary should directly map to
            # the argument names in your GenCastCfdModel.forward() method.
            outputs = model(**batch)
            loss = outputs["loss"]["nmse"] # Or ["mse"] depending on your loss_fn
            loss = loss / gradient_accumulation_steps
        # ... rest of backward pass, optimizer step, etc. ...
    
'''