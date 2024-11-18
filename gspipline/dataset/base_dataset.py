from torch.utils.data import Dataset
from typing import Type, List, Tuple

from nerfstudio.configs.base_config import InstantiateConfig

from dataclasses import dataclass, field


@dataclass
class BaseDatasetConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: BaseDataset)

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return super().setup()


class BaseDataset(Dataset):
    def __init__(self, config: BaseDatasetConfig):
        self.config = config