from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VerificationConfig:
    backend: str = "auto_LiRPA"
    method: str = "CROWN-Optimized"
    device: str = "cpu"
    top_k: int = 20
    input_lower: float = 0.0
    input_upper: float = 1.0
    timeout_sec: Optional[float] = None
    max_logged_split_steps: int = 0
    conv_mode: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetConfig:
    dataset: str
    data_root: str = "./data"
    eps: float = 0.03
    count: int = 10
    start_index: int = 0
    indices: list[int] = field(default_factory=list)
    normalize: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerturbationConfig:
    mode: str = "random_noise"
    random_std: float = 1e-3
    random_relative: bool = True
    quant_bits: int = 8
    prune_fraction: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

