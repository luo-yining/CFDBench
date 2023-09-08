'''
The setting of this problem or datset is as follows.

        ----------------------------




    --->        |
    --->        |
    --->        |
        -----------------------------


Water will run over the dam from left to right.

case_params = {
  "case_no": 0.0,
  "velocity": 0.05,
  "density": 100.0,
  "viscosity": 0.1,
  "barrier_height": 0.1,
  "barrier_width": 0.05,
  "height": 0.4,
  "width": 1.5,
  "dx": 0.0234375,
  "dy": 0.00625
}
'''
from pathlib import Path
from typing import Tuple, List, Dict, Any
import random
from bisect import bisect_right

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from .base import CfdDataset, CfdAutoDataset
from .utils import load_json, normalize_bc, normalize_physics_props


def load_case_data(case_dir: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Load from the file that I have preprocessed, and pad the boundary conditions,
    turn into a numpy array of features.

    The shape of both u and v is (time steps, height, width)
    """
    case_params = load_json(case_dir / "case.json")
    # print(case_params)

    u_file = case_dir / "u.npy"
    v_file = case_dir / "v.npy"
    u = np.load(u_file)
    v = np.load(v_file)
    # Shape of u and v: (time steps, height, width)

    # Mask, 0s for obstacles and walls, 1s for interior.
    mask = np.ones_like(u)
    barrier_width = case_params["barrier_width"]
    barrier_height = case_params["barrier_height"]

    # Set the barrier to zero
    barrier_left = 0.5
    barrier_right = barrier_left + barrier_width
    barrier_bottom = 0
    barrier_top = barrier_height

    barrier_left_idx = int(barrier_left / case_params["dx"])
    barrier_right_idx = int(barrier_right / case_params["dx"])
    barrier_bottom_idx = int(barrier_bottom / case_params["dy"])
    barrier_top_idx = int(barrier_top / case_params["dy"])
    mask[:barrier_bottom_idx:barrier_top_idx, barrier_left_idx:barrier_right_idx] = 0

    # Pad the left side with the BC
    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )
    u[:, :barrier_top_idx, :1] = case_params["velocity"]
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)

    # Pad the top and bottom, which is just 0
    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    features = np.stack([u, v, mask], axis=1)  # (T, 3, h, w)

    # Get wanted case params
    param_keys = ['velocity', 'density', 'viscosity', 'height', 'width']
    case_params = {k: case_params[k] for k in param_keys}
    return features, case_params


class DamFlowDataset(CfdDataset):
    """
    Dataset for Dam flow problem.

    Varying density and viscosity and inlet velocity for each case (3 variables).
    """

    data_delta_time = 0.1  # Time difference (s) between two frames in the data.
    case_params_keys = ['velocity', 'density', 'viscosity', 'dx', 'dy']

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
        - stable_state_diff: The mean relative difference between two consecutive
            frames that indicates the system has reached a stable state.
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
        case_ids: List[int] = []  # 每个样本对应的case的id

        # 遍历每个case的每一帧，构造features和labels
        for case_id, case_dir in enumerate(tqdm(case_dirs)):
            # (T, c, h, w), dict
            this_case_features, this_case_params = load_case_data(case_dir)
            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, 'velocity')

            T, c, h, w = this_case_features.shape
            self.num_features += T * h * w
            params_tensor = torch.tensor(
                [this_case_params[key] for key in self.case_params_keys],
                dtype=torch.float32,
            )
            self.case_params.append(params_tensor)
            features.append(torch.tensor(this_case_features, dtype=torch.float32))
            case_ids.append(case_id)
            self.num_frames.append(T)

        self.features = features  # N * (T, c, h, w)
        self.case_ids = torch.tensor(case_ids)

        # get the no. frames up until this case (inclusive), used for evaluation.
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


class DamFlowAutoDataset(CfdAutoDataset):
    """
    Dataset for Dam flow problem.

    Varying density and viscosity and inlet velocity for each case (3 variables).
    """

    data_delta_time = 0.1  # Time difference (s) between two frames in the data.

    def __init__(
        self,
        case_dirs: List[Path],
        norm_props: bool,
        norm_bc: bool,
        delta_time: float = 0.1,
    ):
        """
        Assume:
        - time: 30s
        - time steps: 120
        - time step size: 0.25s

        Args:
            data_dir: Path to the data directory.
            delta_time: Time step size (in sec) to use for training.
        """
        self.case_dirs = case_dirs
        self.norm_props = norm_props
        self.norm_bc = norm_bc
        self.delta_time = delta_time

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
        # 根据 case ID 来排序 case 子目录
        self.case_params: List[dict] = []
        all_inputs: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_case_ids: List[int] = []  # 每个样本对应的case的id

        # 遍历每个case的每一帧，构造features和labels
        for case_id, case_dir in enumerate(case_dirs):
            case_features, this_case_params = load_case_data(case_dir)  # (T, c, h, w)
            inputs = case_features[:-time_step_size, :]  # (T, 3, h, w)
            outputs = case_features[time_step_size:, :]  # (T, 3, h, w)
            assert len(inputs) == len(outputs)

            if self.norm_props:
                normalize_physics_props(this_case_params)
            if self.norm_bc:
                normalize_bc(this_case_params, "velocity")

            self.case_params.append(this_case_params)
            num_steps = len(outputs)
            # Loop frames, get input-output pairs
            # Stop when converged
            for i in range(num_steps):
                inp = torch.tensor(inputs[i], dtype=torch.float32)  # (2, h, w)
                out = torch.tensor(
                    outputs[i], dtype=torch.float32
                )

                # # Check for convergence
                # inp_magn = torch.sqrt(inp[0] ** 2 + inp[1] ** 2)
                # out_magn = torch.sqrt(out[0] ** 2 + out[1] ** 2)
                # diff = torch.abs(inp_magn - out_magn).mean()
                # # print(f"Mean difference: {diff}")
                # if diff < self.stable_state_diff:
                #     print(f"Converged at {i} out of {num_steps}, {this_case_params}")
                #     break
                assert not torch.isnan(inp).any()
                assert not torch.isnan(out).any()
                all_inputs.append(inp)
                all_labels.append(out[:1])  # Only learn u
                all_case_ids.append(case_id)
        self.inputs = torch.stack(all_inputs)  # (num_samples, 3, h, w)
        self.labels = torch.stack(all_labels)  # (num_samples, 1, h, w)
        self.case_ids = all_case_ids

    def __getitem__(self, idx: int):
        inputs = self.inputs[idx]  # (3, h, w)
        label = self.labels[idx]  # (1, h, w)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id]
        case_params = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in case_params.items()
        }
        return inputs, label, case_params

    def __len__(self):
        return len(self.inputs)


def get_dam_datasets(
    data_dir: Path,
    case_name,
    norm_props: bool,
    norm_bc: bool,
    seed: int = 0,
) -> Tuple[DamFlowDataset, DamFlowDataset, DamFlowDataset]:
    """
    Returns: (train_data, dev_data, test_data)
    """
    case_dirs = []
    for name in ["prop", "bc", "geo"]:
        if name in case_name:
            case_dir = data_dir / name
            this_case_dirs = sorted(
                case_dir.glob("case*"), key=lambda x: int(x.name[4:])
            )
            case_dirs += this_case_dirs
    assert len(case_dirs) > 0
    random.seed(seed)
    random.shuffle(case_dirs)
    # Split into train, dev, test
    num_cases = len(case_dirs)
    num_train = round(num_cases * 0.8)
    num_dev = round(num_cases * 0.1)
    train_case_dirs = case_dirs[:num_train]
    dev_case_dirs = case_dirs[num_train : num_train + num_dev]
    test_case_dirs = case_dirs[num_train + num_dev :]
    train_data = DamFlowDataset(train_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    dev_data = DamFlowDataset(dev_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    test_data = DamFlowDataset(test_case_dirs, norm_props=norm_props, norm_bc=norm_bc)
    return train_data, dev_data, test_data


def get_dam_auto_datasets(
    data_dir: Path,
    case_name: str,
    norm_props: bool,
    norm_bc: bool,
    delta_time: float = 0.1,
    stable_state_diff: float = 0.001,
    seed: int = 0,
) -> Tuple[DamFlowAutoDataset, DamFlowAutoDataset, DamFlowAutoDataset]:
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
        delta_time=delta_time,
        norm_props=norm_props,
        norm_bc=norm_bc,
    )
    train_data = DamFlowAutoDataset(train_case_dirs, **kwargs)
    dev_data = DamFlowAutoDataset(dev_case_dirs, **kwargs)
    test_data = DamFlowAutoDataset(test_case_dirs, **kwargs)
    return train_data, dev_data, test_data


if __name__ == "__main__":
    data_dir = Path("../data/dam/prop")
    dataset = DamFlowDataset([data_dir], norm_props=True, norm_bc=True)
    print(dataset[1])
