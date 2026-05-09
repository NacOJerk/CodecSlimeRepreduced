import os
import pytorch_lightning as pl
import hydra
import torch
import random
import time
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import CodecLightningModule

seed = 1024
seed_everything(seed)


def _build_checkpoint_callback(cfg):
    ckpt_cfg = cfg.train.get("checkpoint", None)
    if ckpt_cfg is None:
        return ModelCheckpoint(
            dirpath=cfg.log_dir, save_top_k=1, save_last=True,
            every_n_train_steps=10000, monitor="mel_loss", mode="min",
        )
    return ModelCheckpoint(
        dirpath=cfg.log_dir,
        save_top_k=ckpt_cfg.get("save_top_k", 1),
        save_last=ckpt_cfg.get("save_last", True),
        every_n_train_steps=ckpt_cfg.get("every_n_train_steps", 10000),
        monitor=ckpt_cfg.get("monitor", "mel_loss"),
        mode=ckpt_cfg.get("mode", "min"),
    )


def _build_logger(cfg):
    logger_cfg = cfg.train.get("logger", None)
    if logger_cfg is None:
        return True
    return hydra.utils.instantiate(logger_cfg)


@hydra.main(config_path="config", config_name="default", version_base=None)
def train(cfg):
    callbacks = [_build_checkpoint_callback(cfg), LearningRateMonitor(logging_interval="step")]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)

    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=_build_logger(cfg),
        limit_train_batches=1.0 if not cfg.debug else 0.001,
    )
    resume_path = cfg.get("ckpt", None)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=resume_path)
    print(f"Training ends, ckpt path: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    train()
