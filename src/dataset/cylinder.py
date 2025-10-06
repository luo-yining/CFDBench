from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from bisect import bisect_right
import random

import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    from base import CfdDataset, CfdAutoDataset
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.common import (
        load_json, normalize_bc, normalize_physics_props)  # type: ignore
else:
    from .base import CfdDataset, CfdAutoDataset
    from .utils import load_json, normalize_bc, normalize_physics_props


def load_case_data(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load from the file that I have preprocessed, and pad the boundary
    conditions, turn into a numpy array of features.

    The shape of both u and v is (time steps, height, width)
    """
    case_params = load_json(case_dir / "case.json")
    # print(case_params)

    u_file = case_dir / "u.npy"
    v_file = case_dir / "v.npy"
    u = np.load(u_file)
    v = np.load(v_file)
    # Shape of u and v: (time steps, height, width)

    # Mask
    mask = np.ones_like(u)
    x_min = case_params["x_min"]
    x_max = case_params["x_max"]
    y_min = case_params["y_min"]
    y_max = case_params["y_max"]
    radius = case_params["radius"]
    case_params["center_x"] = -x_min
    case_params["center_y"] = -y_min
    for key in ["x_min", "x_max", "y_min", "y_max"]:
        del case_params[key]

    # Set the circle to 0
    height = y_max - y_min
    width = x_max - x_min
    case_params["height"] = height
    case_params["width"] = width
    center_x = -x_min
    center_y = -y_min

    dx = width / u.shape[2]
    dy = height / u.shape[1]
    for i in range(u.shape[1]):
        for j in range(u.shape[2]):
            x = x_min + j * dx
            y = y_min + i * dy
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                mask[:, i, j] = 0

    # Pad the left side
    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=case_params["vel_in"],
    )
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    mask = np.pad(
        mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0
    )
    # # Pad the top and bottom
    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(
        mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0
    )
    features = np.stack([u, v, mask], axis=1)  # (T, 3, h, w)
    return features, case_params

import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def load_case_data(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load from the file that I have preprocessed, and pad the boundary
    conditions, turn into a numpy array of features.
    The shape of both u and v is (time steps, height, width)
    
    FIXED VERSION:
    - Applies padding BEFORE mask calculation
    - Uses symmetric padding to get 64x64 output
    - Correctly aligns circle mask with padded coordinates
    """
    case_params = load_json(case_dir / "case.json")
    
    # Load velocity fields
    u_file = case_dir / "u.npy"
    v_file = case_dir / "v.npy"
    u = np.load(u_file)  # Shape: (time_steps, height, width)
    v = np.load(v_file)
    
    # Get domain parameters
    x_min = case_params["x_min"]
    x_max = case_params["x_max"]
    y_min = case_params["y_min"]
    y_max = case_params["y_max"]
    radius = case_params["radius"]
    
    # Store center in case params
    case_params["center_x"] = -x_min
    case_params["center_y"] = -y_min
    
    # Domain dimensions
    height = y_max - y_min
    width = x_max - x_min
    case_params["height"] = height
    case_params["width"] = width
    
    # Clean up redundant params
    for key in ["x_min", "x_max", "y_min", "y_max"]:
        del case_params[key]
    
    # Original dimensions
    original_height = u.shape[1]
    original_width = u.shape[2]
    
    # --- APPLY PADDING FIRST ---
    # Pad left side (inlet boundary condition)
    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=case_params["vel_in"],
    )
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    
    # Pad top and bottom (wall boundary conditions)
    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    
    # Now u and v have shape: (T, original_height + 2, original_width + 1)
    padded_height = u.shape[1]
    padded_width = u.shape[2]
    
    # --- CREATE MASK ON PADDED GRID ---
    mask = np.ones_like(u)  # Start with all valid (1s)
    
    # Padded grid spacing
    dx_padded = width / (padded_width - 1)  # -1 because grid includes boundaries
    dy_padded = height / (padded_height - 1)
    
    # Padded domain extends slightly due to boundary padding
    x_min_padded = x_min - dx_padded  # Left boundary is 1 cell before domain
    y_min_padded = y_min - dy_padded  # Top boundary is 1 cell before domain
    
    center_x = -x_min
    center_y = -y_min
    
    # Create circle mask
    for i in range(padded_height):
        for j in range(padded_width):
            x = x_min_padded + j * dx_padded
            y = y_min_padded + i * dy_padded
            
            # Check if inside circle (obstacle)
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                mask[:, i, j] = 0
    
    # Set boundary cells to 0 (these are BC cells, not interior fluid)
    mask[:, 0, :] = 0   # Top boundary
    mask[:, -1, :] = 0  # Bottom boundary
    mask[:, :, 0] = 0   # Left boundary (inlet)
    # Note: right boundary (outlet) might stay as 1 depending on your BC
    
    # Stack features: (T, 3, H, W)
    features = np.stack([u, v, mask], axis=1)
    
    print(f"Loaded data shape: {features.shape}")
    print(f"  Original grid: {original_height} x {original_width}")
    print(f"  Padded grid: {padded_height} x {padded_width}")
    
    return features, case_params


def load_case_data_fix(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Corrected version: Use cylinder center from case.json, don't calculate it!
    """
    case_params = load_json(case_dir / "case.json")
    
    u = np.load(case_dir / "u.npy")  # Shape: (T, 64, 64)
    v = np.load(case_dir / "v.npy")
    
    print(f"Original data shape: u={u.shape}, v={v.shape}")
    
    # Get domain bounds
    x_min = case_params["x_min"]
    x_max = case_params["x_max"]
    y_min = case_params["y_min"]
    y_max = case_params["y_max"]
    radius = case_params["radius"]
    
    # Get cylinder center from case params (should already be in case.json!)
    # If not present, use standard CFD benchmark position
    if "center_x" in case_params and "center_y" in case_params:
        center_x = case_params["center_x"]
        center_y = case_params["center_y"]
        print(f"Using cylinder center from case.json: ({center_x}, {center_y})")
    else:
        # Cylinder is at the origin (0, 0) of the coordinate system
        center_x = 0.0
        center_y = 0.0
        case_params["center_x"] = center_x
        case_params["center_y"] = center_y
        print(f"Using default cylinder center: ({center_x:.6f}, {center_y:.6f})")
    
    # Physical domain size
    height = y_max - y_min
    width = x_max - x_min
    case_params["height"] = height
    case_params["width"] = width
    
    # Clean up domain bounds (keep center!)
    for key in ["x_min", "x_max", "y_min", "y_max"]:
        if key in case_params:
            del case_params[key]
    
    # Create mask on 64x64 grid
    mask = np.ones_like(u)
    grid_height, grid_width = u.shape[1], u.shape[2]
    
    # Grid spacing in physical units
    dx = width / grid_width
    dy = height / grid_height
    
    print(f"Physical domain: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"Grid: {grid_height} x {grid_width}, spacing: dx={dx:.5f}, dy={dy:.5f}")
    print(f"Cylinder: center=({center_x:.3f}, {center_y:.3f}), radius={radius:.3f}")
    
    # Calculate cylinder mask
    cylinder_points = 0
    for i in range(grid_height):
        for j in range(grid_width):
            # Physical coordinates of grid cell center
            x = x_min + (j + 0.5) * dx
            y = y_min + (i + 0.5) * dy
            
            # Distance from cylinder center
            dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
            
            if dist_sq <= radius ** 2:
                mask[:, i, j] = 0
                cylinder_points += 1
    
    print(f"Cylinder mask: {cylinder_points} grid points")
    
    if cylinder_points == 0:
        print("\n⚠️  WARNING: No cylinder points found! Cylinder is outside domain.")
        print(f"   Domain X: [{x_min:.3f}, {x_max:.3f}], Cylinder X: {center_x:.3f}")
        print(f"   Domain Y: [{y_min:.3f}, {y_max:.3f}], Cylinder Y: {center_y:.3f}")
    
    # Set boundary cells to 0 (boundary conditions)
    mask[:, 0, :] = 0    # Top boundary
    mask[:, -1, :] = 0   # Bottom boundary  
    mask[:, :, 0] = 0    # Left boundary (inlet)
    # mask[:, :, -1] = 0  # Right boundary (outlet) - uncomment if needed
    
    features = np.stack([u, v, mask], axis=1)
    
    print(f"Output shape: {features.shape}")
    print(f"Mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean():.3f}\n")
    
    return features, case_params


class CylinderFlowDataset(CfdDataset):
    """
    Dataset for Cylinder flow problem.

    Varying density and viscosity and inlet velocity for each case
    (3 variables).
    """

    data_delta_time = (
        0.1  # Time difference (s) between two frames in the data.
    )
    data_max_time = 30  # Total time (s) in the data.
    case_params_keys = [
        "vel_in",
        "density",
        "viscosity",
        "height",
        "width",
        "center_x",
        "center_y",
        "radius",
    ]

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        sample_point_by_point: bool = False,
        stable_state_diff: float = 0.001,
    ):
        """
        Args:
        - data_dir:
        - norm_props: whether to normalize physics properties.
        - sample_point_by_point: If True, each example is a feature point
            (x, y, t) and the corresponding output function value u(x, y, t).
            If False, each example is an entire frame.
        - stable_state_diff: The mean relative difference between two
            consecutive frames that indicates the system has reached a
            stable state.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.sample_point_by_point = sample_point_by_point
        self.stable_state_diff = stable_state_diff

        self.load_data(case_dirs)

    def load_data(self, case_dirs: List[Path]):
        """
        This will set the following attributes:
            self.case_params: List[dict]
            self.features: List[Tensor]  # (N, T, 2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        where N is the number of cases.
        """
        self.case_params: List[Tensor] = []
        self.num_features = 0
        self.num_frames: List[int] = []
        features: List[Tensor] = []
        case_ids: List[int] = [] 

    
        for case_id, case_dir in enumerate(tqdm(case_dirs)):
            # (T, c, h, w), dict
            this_case_features, this_case_params = load_case_data_fix(case_dir)
            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_in")

            T, c, h, w = this_case_features.shape
            self.num_features += T * h * w
            params_tensor = torch.tensor(
                [this_case_params[key] for key in self.case_params_keys],
                dtype=torch.float32,
            )
            self.case_params.append(params_tensor)
            features.append(
                torch.tensor(this_case_features, dtype=torch.float32)
            )
            case_ids.append(case_id)
            self.num_frames.append(T)

        self.features = features  # N * (T, c, h, w)
        self.case_ids = torch.tensor(case_ids)

        # get the no. frames up until this case (inclusive), used for
        # evaluation.
        self.num_frames_before: List[int] = [
            sum(self.num_frames[: i + 1]) for i in range(len(self.num_frames))
        ]

    def idx_to_case_id_and_frame_idx(self, idx: int) -> Tuple[int, int]:
        """
        Given an index, return the case ID of the corresponding example.
        Will be using `self.num_frames_before`.

        For instance, if the number of frames in the first three cases are
        [10, 12, 11], then:
        - 0~9 should map to case_id = 0
        - 10~21 should map to case_id = 1
        - 22~32 should map to case_id = 2
        In this case, `num_frames_before` should be [10, 22, 33].
        """
        case_id = bisect_right(self.num_frames_before, idx)
        if case_id == 0:
            frame_idx = idx
        else:
            frame_idx = idx - self.num_frames_before[case_id - 1]
        return case_id, frame_idx

    def __getitem__(self, idx: int):
        # During evaluation, we need an entire frame
        # So each example returns (case_params, frame)
        # The number of examples is
        case_id, frame_idx = self.idx_to_case_id_and_frame_idx(idx)
        t = torch.tensor([frame_idx]).float()
        frame = self.features[case_id][frame_idx]  # (T, c, h, w)
        case_params = self.case_params[case_id]
        return case_params, t, frame

    def __len__(self):
        return self.num_frames_before[-1]


class CylinderFlowAutoDataset(CfdAutoDataset):
    """
    Dataset for Laminar flow problem.

    Varying density and viscosity and inlet velocity for each case
    (3 variables).
    """

    data_delta_time = (
        0.001  # Time difference (s) between two frames in the data.
    )
    # data_max_time = 30  # Total time (s) in the data.

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        split: str,
        cache_dir: Path,
        delta_time: float = 0.1,
        stable_state_diff: float = 0.001,
    ):
        """
        Assume:
        - time: 30s
        - time steps: 120
        - time step size: 0.25s

        geometry 0：d=0.1m，l=1m
        geometry 1-10 ：d=0.05-0.09m,0.11-0.15m，l=1m
        geometry 11-20 ：d=0.1m，l=0.5-0.95m

        case0：入口速度0.1 m/s，密度1000 kg /m^3，动力粘度0.01 Pa-s
        case1-case21：入口速度0.05-0.15 m/s，du=0.005 m/s
        case22-case42：密度900-1100 kg /m^3，dρ=10 kg/m^3
        case43-case63：动力粘度0.005-0.015 Pa-s，dv=0.0005 Pa-s

        Args:
            data_dir: Path to the data directory.
            delta_time: Time step size (in sec) to use for training.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.split = split
        self.cache_dir = cache_dir / split
        self.delta_time = delta_time
        self.stable_state_diff = stable_state_diff

        # The difference between input and output in number of frames.
        self.time_step_size = int(self.delta_time / self.data_delta_time)
        self.load_data(case_dirs, self.time_step_size)

    def load_data(self, case_dirs, time_step_size: int):
        """
        This will set the following attributes:
            self.case_dirs: List[Path]
            self.case_params: List[dict]
            self.inputs: List[Tensor]  # (2, h, w)
            self.labels: List[Tensor]  # (2, h, w)
            self.case_ids: List[int]  # Each sample's case ID
        """
        # Cache preprocessing for speedup
        if self.cache_dir.exists():
            print(f"Loading from cache: {self.cache_dir}")
            self.inputs = torch.load(self.cache_dir / "inputs.pt", weights_only=False)
            self.labels = torch.load(self.cache_dir / "labels.pt", weights_only=False)
            self.case_ids = torch.load(self.cache_dir / "case_ids.pt", weights_only=False)
            self.case_params = torch.load(self.cache_dir / "case_params.pt", weights_only=False)
            self.all_features = torch.load(self.cache_dir / "all_features.pt", weights_only=False)
            return

        self.case_params: List[dict] = []
        all_inputs: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_case_ids: List[int] = []  # The case ID of each example
        self.all_features: List[np.ndarray] = []

        # Loop cases to create features and labels
        for case_id, case_dir in enumerate(case_dirs):
            case_features, this_case_params = load_case_data_fix(
                case_dir
            )  # (T, c, h, w)
            self.all_features.append(case_features)
            inputs = case_features[:-time_step_size, :]  # (T, 3, h, w)
            outputs = case_features[time_step_size:, :]  # (T, 3, h, w)
            assert len(inputs) == len(outputs)

            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "vel_in")

            self.case_params.append(this_case_params)
            num_steps = len(outputs)
            # Loop frames, get input-output pairs
            # Stop when converged
            for i in range(num_steps):
                inp = torch.tensor(inputs[i], dtype=torch.float32)  # (2, h, w)
                out = torch.tensor(outputs[i], dtype=torch.float32)

                # Check for convergence
                inp_magn = torch.sqrt(inp[0] ** 2 + inp[1] ** 2)
                out_magn = torch.sqrt(out[0] ** 2 + out[1] ** 2)
                diff = torch.abs(inp_magn - out_magn).mean()
                # print(f"Mean difference: {diff}")
                if diff < self.stable_state_diff:
                    print(
                        f"Converged at {i} / {num_steps}, {this_case_params}"
                    )
                    break
                assert not torch.isnan(inp).any()
                assert not torch.isnan(out).any()
                all_inputs.append(inp)
                all_labels.append(out)
                all_case_ids.append(case_id)

        self.inputs = torch.stack(all_inputs)  # (num_samples, 3, h, w)
        self.labels = torch.stack(all_labels)  # (num_samples, 1, h, w)
        self.case_ids = all_case_ids

        # Cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.inputs, self.cache_dir / "inputs.pt")
        torch.save(self.labels, self.cache_dir / "labels.pt")
        torch.save(self.case_ids, self.cache_dir / "case_ids.pt")
        torch.save(self.case_params, self.cache_dir / "case_params.pt")
        torch.save(self.all_features, self.cache_dir / "all_features.pt")

    def __getitem__(self, idx: int):
        inputs = self.inputs[idx]  # (3, h, w)
        label = self.labels[idx]  # (1, h, w)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id]
        case_params = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in case_params.items()
        }
        # print(inputs.shape)
        # print(label.shape)
        return inputs, label, case_params

    def __len__(self):
        return len(self.inputs)


def get_cylinder_datasets(
    data_dir: Path,
    case_name: str,
    norm_props: bool,
    norm_bc: bool,
    seed: int = 0,
) -> Tuple[CfdDataset, CfdDataset, CfdDataset]:
    print(data_dir, case_name)
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in case_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs

    assert case_dirs

    random.seed(seed)
    random.shuffle(case_dirs)

    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    print("==== Number of cases in different splits ====")
    print(
        f"train: {len(train_case_dirs)}, "
        f"dev: {len(dev_case_dirs)}, "
        f"test: {len(test_case_dirs)}"
    )
    print("=============================================")
    kwargs: dict[str, Any] = dict(
        norm_props=norm_props,
        norm_bc=norm_bc,
    )
    train_data = CylinderFlowDataset(train_case_dirs, **kwargs)
    dev_data = CylinderFlowDataset(dev_case_dirs, **kwargs)
    test_data = CylinderFlowDataset(test_case_dirs, **kwargs)
    return train_data, dev_data, test_data


def get_cylinder_auto_datasets(
    data_dir: Path,
    subset_name: str,
    norm_props: bool,
    norm_bc: bool,
    delta_time: float = 0.01,
    stable_state_diff: float = 0.001,
    seed: int = 0,
    load_splits: List[str] = ["train", "dev", "test"],
) -> Tuple[
    Optional[CylinderFlowAutoDataset],
    Optional[CylinderFlowAutoDataset],
    Optional[CylinderFlowAutoDataset],
]:
    print(data_dir, subset_name)
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in subset_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs

    assert case_dirs

    random.seed(seed)
    random.shuffle(case_dirs)

    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = int(num_cases * 0.8)
    num_dev = int(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    print("==== Number of cases in different splits ====")
    print(
        f"train: {len(train_case_dirs)}, "
        f"dev: {len(dev_case_dirs)}, "
        f"test: {len(test_case_dirs)}"
    )
    print("=============================================")
    kwargs: dict[str, Any] = dict(
        delta_time=delta_time,
        stable_state_diff=stable_state_diff,
        norm_props=norm_props,
        norm_bc=norm_bc,
        cache_dir=Path("./dataset/cache/cylinder", subset_name),
    )
    if "train" in load_splits:
        train_data = CylinderFlowAutoDataset(
            train_case_dirs, split="train", **kwargs
        )
    else:
        train_data = None
    if "dev" in load_splits:
        dev_data = CylinderFlowAutoDataset(
            dev_case_dirs, split="dev", **kwargs
        )
    else:
        dev_data = None
    if "test" in load_splits:
        test_data = CylinderFlowAutoDataset(
            test_case_dirs, split="test", **kwargs
        )
    else:
        test_data = None
    return train_data, dev_data, test_data


if __name__ == "__main__":
    case_dir = Path("../../data/cylinder/prop/case0070")
    dataset = CylinderFlowDataset([case_dir], norm_props=True, norm_bc=True)
    inputs, labels, case_params = dataset[1]
    print("nonautoregressive example")
    print(inputs.shape, labels.shape)
    print(case_params)

    dataset = CylinderFlowAutoDataset(
        [case_dir],
        norm_props=True,
        norm_bc=True,
        cache_dir=Path("./cache/cylinder/test"),
        split="test",
    )
    inputs, labels, case_params = dataset[1]
    print("autoregressive example")
    print(inputs.shape, labels.shape)
    print(case_params)
