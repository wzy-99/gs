import random
import numpy as np

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch import Generator

from lightning.pytorch import LightningDataModule

from gspipline.dataset.base_dataset import BaseDatasetConfig

from dataclasses import dataclass, field


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    """Batch size for the data loader."""
    num_workers: int = 0
    """Number of workers to use for the data loader."""
    pin_memory: bool = False
    """Whether to pin memory for the data loader."""
    drop_last: bool = False
    """Whether to drop the last incomplete batch."""
    seed: int | None = None
    """Random seed for the data loader."""
    persistent_workers: bool = False
    """Whether to use persistent workers for the data loader."""


@dataclass
class DataModuleConfig:
    train: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """Configuration for the training data loader."""
    val: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """Configuration for the validation data loader."""
    test: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    """Configuration for the test data loader."""


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    config: DataModuleConfig

    def __init__(
        self,
        config: DataModuleConfig,
        dataset_configs: list[BaseDatasetConfig]
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset_configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs]
        self.global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def get_persistent(self, loader_cfg: DataLoaderConfig) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderConfig) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        datasets = [dataset_config.setup(stage='train') for dataset_config in self.dataset_configs]
        data_loaders = []
        for dataset in datasets:
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.config.train.batch_size,
                    shuffle=not isinstance(dataset, IterableDataset),
                    num_workers=self.config.train.num_workers,
                    generator=self.get_generator(self.config.train),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.config.train),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def val_dataloader(self):
        datasets = [dataset_config.setup(stage='val') for dataset_config in self.dataset_configs]
        data_loaders = []
        for dataset in datasets:
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.config.val.batch_size,
                    num_workers=self.config.val.num_workers,
                    generator=self.get_generator(self.config.val),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.config.val),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def test_dataloader(self):
        datasets = [dataset_config.setup(stage='test') for dataset_config in self.dataset_configs]
        data_loaders = []
        for dataset in datasets:
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.config.test.batch_size,
                    num_workers=self.config.test.num_workers,
                    generator=self.get_generator(self.config.test),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(self.config.test),
                )
            )
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]