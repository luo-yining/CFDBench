from pathlib import Path
from typing import Tuple

from .base import CfdDataset, CfdAutoDataset
from .tube import get_tube_datasets, get_tube_auto_datasets
from .cavity import get_cavity_datasets, get_cavity_auto_datasets
from .cylinder import get_cylinder_datasets, get_cylinder_auto_datasets
from .dam import get_dam_datasets, get_dam_auto_datasets


def get_dataset(
    data_name: str,
    data_dir: Path,
    norm_props: bool,
    norm_bc: bool,
) -> Tuple[CfdDataset, CfdDataset, CfdDataset]:
    """
    Args:
        data_name: One of: "cavity", "tube", "dam", "cylinder"
        time_step_size: The time difference between input and output.
    """
    problem_name = data_name.split("_")[0]
    subset_name = data_name[len(problem_name) + 1 :]
    assert problem_name in ["cavity", "tube", "dam", "cylinder"]
    print(f"Loading problem: {problem_name}, subset: {subset_name}")
    if problem_name == "tube":
        train_data, dev_data, test_data = get_tube_datasets(
            data_dir / problem_name,
            subset_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif problem_name == "cavity":
        # data_dir = Path("data", data_name)
        train_data, dev_data, test_data = get_cavity_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif problem_name == "dam":
        train_data, dev_data, test_data = get_dam_datasets(
            data_dir / problem_name,
            case_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif problem_name == 'cylinder':
        train_data, dev_data, test_data = get_cylinder_datasets(
            data_dir / problem_name,
            case_name=subset_name,
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
        data_name: One of: "cavity", "tube", "dam", "cylinder"
        delta_time: The time difference between input and output.
    """
    problem_name = data_name.split("_")[0]
    assert problem_name in ["cavity", "tube", "dam", "cylinder"]
    subset_name = data_name[len(problem_name) + 1 :]
    assert delta_time > 0
    print("Loading data...")
    if problem_name == "tube":
        train_data, dev_data, test_data = get_tube_auto_datasets(
            data_dir / problem_name,
            subset_name=subset_name,
            delta_time=delta_time,
            norm_props=norm_props,
            norm_bc=norm_bc,
        )
        return train_data, dev_data, test_data
    elif problem_name == "cavity":
        train_data, dev_data, test_data = get_cavity_auto_datasets(
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
            subset_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
            delta_time=delta_time,
        )
        return train_data, dev_data, test_data
    elif problem_name == 'cylinder':
        train_data, dev_data, test_data = get_cylinder_auto_datasets(
            data_dir / problem_name,
            subset_name=subset_name,
            norm_props=norm_props,
            norm_bc=norm_bc,
            delta_time=delta_time,
        )
        return train_data, dev_data, test_data
    else:
        raise ValueError(f"Invalid data name \"{data_name}\"")
