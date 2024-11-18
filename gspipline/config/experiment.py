from dataclasses import dataclass, field
from typing import List, Union, Dict, Any, Tuple, Optional, Literal

from gspipline.dataset.data_module import DataModuleConfig


@dataclass
class TrainerConfig:
    max_epochs: int = -1
    """Maximum number of epochs to train the model. If set to -1, the model will train indefinitely."""
    max_steps: int = 100000
    """Maximum number of steps to train the model. If set to -1, the model will train indefinitely."""
    num_nodes: int = 1
    """Number of nodes to use for distributed training."""
    accelerator: str = "gpu"
    """Accelerator to use for distributed training. Can be "gpu" or "tpu"."""
    devices: str | List[int] | int = "auto"
    """Devices to use for distributed training. Can be "auto" or a list of device IDs."""
    strategy: str = "ddp_find_unused_parameters_true"
    """Strategy to use for distributed training. Can be "ddp" or "dp"."""
    val_check_interval: float | int = 1.0
    """How often to check the validation set during training. Set to 1.0 to check every epoch."""
    check_val_every_n_epoch: int = 1
    """How often to check the validation set during training. Set to 1 to check every epoch."""
    accumulate_grad_batches: int = 1
    """Number of batches to accumulate before performing a backward/update pass."""
    gradient_clip_val: float | None = None
    """Value to clip the gradients to. Set to None to disable gradient clipping."""
    fast_dev_run: bool | int = False
    """Whether to run a fast development run. If True, only a few batches of data will be used."""
    limit_train_batches: float | int = 1.0
    """Fraction of the training set to use for training. Set to 1.0 to use the entire training set."""
    limit_val_batches: float | int = 1.0
    """Fraction of the validation set to use for validation. Set to 1.0 to use the entire validation set."""
    limit_test_batches: float | int = 1.0
    """Fraction of the test set to use for testing. Set to 1.0 to use the entire test set."""
    num_sanity_val_steps: int = 0
    """Number of validation steps to run before starting the training loop."""
    overfit_batches : float | int = 0.0
    """Fraction of the training set to use for overfitting. Set to 0.0 to disable overfitting."""
    precision: int = 16
    """Precision to use for training. Can be 16 or 32."""
    profiler: str | None = None
    """Profiler to use for profiling the training process. Can be "simple", "advanced", or None."""
    enable_progress_bar: bool = True
    """Whether to enable the progress bar during training."""
    log_every_n_steps: int = 50
    """How often to log the training process."""
    logger: Literal["tensorboard", "wandb", "csv", "mlflow", "comet", "neptune", "sacred", "testtube", "none"] = "tensorboard"


@dataclass
class CheckpointConfig:
    every_n_train_steps: int = 1000
    """How often to save checkpoints during training."""
    save_top_k: int = 1
    """Number of top checkpoints to save."""
    save_weights_only: bool = True
    """Whether to save only the model weights."""


@dataclass  
class ExperimentConfig:
    seed: int = 42
    """Random seed for reproducibility."""
    mode: Literal["train", "test", "predict"] = "train"
    """Mode of the experiment. Can be "train", "test", or "predict"."""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Configuration for the trainer."""
    data_module: DataModuleConfig = field(default_factory=DataModuleConfig)
    """Configuration for the data loaders."""
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Configuration for the checkpoints."""
    name: str = "test"
    """Name of the experiment."""
    project: str = "gspipline"
    """Project name for logging."""