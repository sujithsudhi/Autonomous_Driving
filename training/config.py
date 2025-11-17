"""Structured configuration utilities for BEVFormer-Lite training."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Experiment naming and output directory settings."""

    name: str
    output_dir: str


@dataclass
class DatasetConfig:
    """NuScenes dataset parameters."""

    root: str
    version: str
    cameras: List[str]
    image_size: List[int]
    train_split: str = "train"
    val_split: str = "val"
    bev_bounds: Dict[str, List[float]] = field(default_factory=dict)
    bev_resolution: float = 1.0


@dataclass
class ModelConfig:
    """Model backbone and architecture choices."""

    backbone: str
    embed_dim: int
    bev_encoder_layers: int
    bev_num_heads: int
    cross_attn_chunk_size: int = 256
    max_cross_attn_elements: int = 25_000_000
    camera_token_stride: int = 1
    dropout: float = 0.0
    num_classes: int = 10


@dataclass
class OptimizationConfig:
    """Training hyperparameters."""

    epochs: int
    batch_size: int
    val_batch_size: Optional[int] = None
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 0
    num_workers: int = 4
    val_num_workers: Optional[int] = None
    mixed_precision: bool = True
    lr_plateau_patience: int = 5
    lr_reduce_factor: float = 0.5
    early_stop_patience: int = 10
    max_grad_norm: float = 5.0


@dataclass
class LossConfig:
    """Loss weighting parameters."""

    cls_weight: float = 1.0
    box_weight: float = 1.0


@dataclass
class LoggingConfig:
    """Console/logging settings."""

    print_interval: int = 20


@dataclass
class TrainingConfig:
    """Structured config matching the Transformers-style YAML layout."""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimization: OptimizationConfig
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            experiment=ExperimentConfig(**data["experiment"]),
            dataset=DatasetConfig(**data["dataset"]),
            model=ModelConfig(**data["model"]),
            optimization=OptimizationConfig(**data["optimization"]),
            loss=LossConfig(**data.get("loss", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

