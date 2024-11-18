import os
from pathlib import Path

import hydra
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from gspipline.pipeline.gaussian_discriminator import GaussianDiscriminatorPipeline, GaussianDiscriminatorPipelineConfig
from gspipline.dataset.nuscene_novel_view import nuSceneNovelViewConfig, nuSceneNovelViewDataset
from gspipline.dataset.data_module import DataModule
from gspipline.config.experiment import ExperimentConfig

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field


@dataclass
class GaussianDiscriminatorExperimentConfig(ExperimentConfig):
    pipeline: GaussianDiscriminatorPipelineConfig = field(default_factory=GaussianDiscriminatorPipelineConfig)
    dataset: nuSceneNovelViewConfig = field(default_factory=nuSceneNovelViewConfig)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="gaussian_discriminator",
)
def train(cfg: DictConfig):

    cfg: GaussianDiscriminatorExperimentConfig = from_dict(data_class=GaussianDiscriminatorExperimentConfig, 
                                                           data=OmegaConf.to_container(cfg), 
                                                           config=Config(type_hooks={Path: Path, tuple[int, int]: tuple}))

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )

    callbacks = []
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor=cfg.checkpointing.monitor,
            mode="max",
        )
    )

    if cfg.trainer.logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            name=cfg.name,
            project=cfg.project,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg),
        )
    elif cfg.trainer.logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name=cfg.name,
        )
    else:
        logger = None

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        fast_dev_run=cfg.trainer.fast_dev_run,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        overfit_batches=cfg.trainer.overfit_batches,
        precision=cfg.trainer.precision,
        profiler=cfg.trainer.profiler,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if cfg.mode == "test" else True,
    )
    seed_everything(cfg.seed + trainer.global_rank)

    pipeline = GaussianDiscriminatorPipeline(cfg.pipeline)

    data_module = DataModule(cfg.data_module, cfg.dataset)

    checkpoint_path = None

    if cfg.mode == "train":
        trainer.fit(pipeline, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            pipeline,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
