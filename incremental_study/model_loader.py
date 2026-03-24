from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "incremental_study" / "vendor"


DATASET_SPECS = {
    "mnist": {"in_ch": 1, "in_dim": 28, "input_shape": (1, 28, 28), "num_classes": 10},
    "cifar10": {"in_ch": 3, "in_dim": 32, "input_shape": (3, 32, 32), "num_classes": 10},
}


@dataclass
class LoadedModel:
    model: nn.Module
    model_path: str
    architecture: str
    dataset: str
    input_shape: tuple[int, ...]
    num_classes: int
    source_format: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "architecture": self.architecture,
            "dataset": self.dataset,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "source_format": self.source_format,
        }


class NormalizationWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def _dataset_key(dataset: str) -> str:
    key = dataset.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return key


def _ensure_vendor_importable() -> None:
    vendor_root = str(VENDOR_ROOT)
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)


def _dataset_normalization(dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    key = _dataset_key(dataset)
    if key == "mnist":
        mean = torch.tensor([0.0]).view(1, 1, 1)
        std = torch.tensor([1.0]).view(1, 1, 1)
    elif key == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return mean, std


def _extract_state_dict(payload: Any, model_path: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "state_dict" in payload:
            state_dict = payload["state_dict"]
            if isinstance(state_dict, (list, tuple)):
                if not state_dict:
                    raise ValueError(f"Empty state_dict list in {model_path}")
                state_dict = state_dict[0]
            if isinstance(state_dict, dict):
                return state_dict
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError(f"Unsupported checkpoint format for {model_path}")


def _infer_architecture_name(model_path: str, explicit_architecture: Optional[str]) -> str:
    if explicit_architecture:
        return explicit_architecture
    return Path(model_path).stem


def _instantiate_ivan_architecture(architecture: str, dataset: str) -> nn.Module:
    _ensure_vendor_importable()
    from . import models as study_models

    if architecture not in study_models.Models:
        raise KeyError(
            f"Architecture '{architecture}' is not present in the vendored model registry. "
            f"Pass --model-arch explicitly if the checkpoint stem does not match."
        )

    ctor = study_models.Models[architecture]
    params = inspect.signature(ctor).parameters
    spec = DATASET_SPECS[_dataset_key(dataset)]
    kwargs: dict[str, Any] = {}
    if "in_ch" in params:
        kwargs["in_ch"] = spec["in_ch"]
    if "in_dim" in params:
        kwargs["in_dim"] = spec["in_dim"]
    return ctor(**kwargs)


def _load_torch_checkpoint(model_path: str, dataset: str, architecture: Optional[str]) -> LoadedModel:
    arch = _infer_architecture_name(model_path, architecture)
    model = _instantiate_ivan_architecture(arch, dataset)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint, model_path)
    model.load_state_dict(state_dict)
    model.eval()
    spec = DATASET_SPECS[_dataset_key(dataset)]
    return LoadedModel(
        model=model,
        model_path=str(Path(model_path).resolve()),
        architecture=arch,
        dataset=_dataset_key(dataset),
        input_shape=spec["input_shape"],
        num_classes=spec["num_classes"],
        source_format="torch",
    )


def _infer_onnx_input_shape(onnx_model: Any) -> tuple[int, ...]:
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    feed_inputs = [node for node in onnx_model.graph.input if node.name in set(input_all) - set(input_initializer)]
    if not feed_inputs:
        feed_inputs = [onnx_model.graph.input[0]]
    dims = feed_inputs[0].type.tensor_type.shape.dim
    values = [int(dim.dim_value) for dim in dims]
    if values and values[0] in (0, 1):
        values = values[1:]
    return tuple(values)


def _load_onnx_model(model_path: str, dataset: str) -> LoadedModel:
    try:
        import onnx  # type: ignore
        from onnx2pytorch import ConvertModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ONNX model loading requires 'onnx' and 'onnx2pytorch' in the runtime environment."
        ) from exc

    onnx_model = onnx.load(model_path)
    model = ConvertModel(onnx_model, experimental=False, debug=False)
    model.eval()
    input_shape = _infer_onnx_input_shape(onnx_model)
    num_classes = DATASET_SPECS[_dataset_key(dataset)]["num_classes"]
    return LoadedModel(
        model=model,
        model_path=str(Path(model_path).resolve()),
        architecture=Path(model_path).stem,
        dataset=_dataset_key(dataset),
        input_shape=input_shape,
        num_classes=num_classes,
        source_format="onnx",
    )


def load_model(
    model_path: str,
    dataset: str,
    *,
    architecture: Optional[str] = None,
    normalization: str = "none",
) -> LoadedModel:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pth", ".pt"}:
        loaded = _load_torch_checkpoint(str(path), dataset, architecture)
    elif suffix == ".onnx":
        loaded = _load_onnx_model(str(path), dataset)
    else:
        raise ValueError(f"Unsupported model format: {suffix}")

    if normalization == "ivan-default":
        mean, std = _dataset_normalization(dataset)
        loaded.model = NormalizationWrapper(loaded.model, mean, std)
    elif normalization != "none":
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    loaded.model.eval()
    return loaded
