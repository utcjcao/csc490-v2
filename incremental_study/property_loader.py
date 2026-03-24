from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torchvision
import torchvision.transforms as transforms


@dataclass
class RobustnessProperty:
    property_id: str
    property_index: int
    dataset_index: int
    label: int
    eps: float
    x: torch.Tensor

    def metadata(self) -> dict:
        return {
            "property_id": self.property_id,
            "property_index": self.property_index,
            "dataset_index": self.dataset_index,
            "label": self.label,
            "epsilon": self.eps,
        }


def _dataset_key(dataset: str) -> str:
    key = dataset.lower()
    if key not in {"mnist", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return key


def _load_dataset(dataset: str, data_root: str):
    transform = transforms.Compose([transforms.ToTensor()])
    key = _dataset_key(dataset)
    if key == "mnist":
        return torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    if key == "cifar10":
        return torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _resolve_indices(count: int, start_index: int, indices: Iterable[int] | None) -> list[int]:
    if indices:
        return [int(idx) for idx in indices]
    return list(range(int(start_index), int(start_index) + int(count)))


def load_properties(
    dataset: str,
    data_root: str,
    *,
    eps: float,
    count: int,
    start_index: int = 0,
    indices: Iterable[int] | None = None,
) -> list[RobustnessProperty]:
    dataset_obj = _load_dataset(dataset, data_root)
    selected_indices = _resolve_indices(count=count, start_index=start_index, indices=indices)

    props: list[RobustnessProperty] = []
    for prop_idx, dataset_idx in enumerate(selected_indices):
        x, label = dataset_obj[dataset_idx]
        label_int = int(label)
        props.append(
            RobustnessProperty(
                property_id=f"{dataset.lower()}_{dataset_idx}_eps_{eps:g}",
                property_index=prop_idx,
                dataset_index=dataset_idx,
                label=label_int,
                eps=float(eps),
                x=x.detach().clone(),
            )
        )
    return props

