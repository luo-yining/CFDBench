from pathlib import Path
from typing import Tuple

from .base import CfdDataset, CfdAutoDataset
from .laminar import get_laminar_datasets, get_laminar_auto_datasets
from .cavity import get_cavity_datasets, get_cavity_auto_datasets
from .karman import get_karman_datasets, get_karman_auto_datasets
from .dam import get_dam_datasets, get_dam_auto_datasets


def get_dataset(
    data_name: str,
    data_dir: Path,
    norm_props: bool,
    norm_bc: bool,
) -> Tuple[CfdDataset, CfdDataset, CfdDataset]:
    """
    Args:
        data_name: One of: 'poiseuille', 'cavity', 'karman', 'dam'
        time_step_size: The time difference between input and output.
    """
    data_name = data_name.split("_")[0]
    data_name = data_name[len(data_name) + 1 :]
    assert data_name in ["laminar", "cavity", "karman", "dam"]
    print("Loading data...")
    if data_name == "laminar":
        train_data, dev_data, test_data = get_laminar_datasets(
            data_dir / data_name,
            data_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif data_name == "cavity":
        # data_dir = Path("data", data_name)
        train_data, dev_data, test_data = get_cavity_datasets(
            data_dir / data_name,
            case_name=data_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif data_name == 'karman':
        train_data, dev_data, test_data = get_karman_datasets(
            data_dir / data_name,
            case_name=data_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif data_name == "dam":
        train_data, dev_data, test_data = get_dam_datasets(
            data_dir / data_name,
            case_name=data_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    else:
        raise ValueError(f"Invalid data name {data_name}!")


def get_auto_dataset(
    data_dir: Path,
    data_name: str,
    delta_time: float,
    norm_props: bool,
    norm_bc: bool,
) -> Tuple[CfdAutoDataset, CfdAutoDataset, CfdAutoDataset]:
    """
    Args:
        data_name: One of: 'poiseuille', 'cavity', 'karman'
        delta_time: The time difference between input and output.
    """
    problem_name = data_name.split("_")[0]
    assert problem_name in ["cavity", "laminar", "cylinder", "dam"]
    subset_name = data_name[len(problem_name) + 1 :]
    assert delta_time > 0
    print("Loading data...")
    if problem_name == "laminar":
        train_data, dev_data, test_data = get_laminar_auto_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            delta_time=delta_time,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif problem_name == "cavity":
        # data_dir = Path("data", data_name)
        train_data, dev_data, test_data = get_cavity_auto_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
            delta_time=delta_time,
        )
        return train_data, dev_data, test_data
    elif problem_name == 'karman':
        train_data, dev_data, test_data = get_karman_auto_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
            delta_time=delta_time,
        )
        return train_data, dev_data, test_data
    elif problem_name == "dam":
        train_data, dev_data, test_data = get_dam_auto_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
            delta_time=delta_time,
        )
        return train_data, dev_data, test_data
    else:
        raise ValueError(f"Invalid data name {data_name}!")
