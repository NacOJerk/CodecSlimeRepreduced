"""Melt-stage training entry point.

The Hydra config (backbones/configs/codecslime_melt_{vq8k,fsq18k}.yaml) selects
backbone, dataset, and Melt hyperparameters. On a fresh launch we load weights
from `cfg.backbone_ckpt` into `CoolMeltWrapper` and start training with a fresh
optimizer / scheduler. On resume (`cfg.ckpt` set), we let Lightning restore
both model and optimizer state from the Melt checkpoint.
"""
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))

from data_module import DataModule
from melt_manager import MeltManager
from cool_manager import CoolManager
from melt_wrapper import CoolMeltWrapper


seed_everything(1024)


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


def _build_compression(cfg):
    kind = cfg.train.compression.kind
    if kind == "melt":
        mm = MeltManager(
            max_compression=cfg.train.compression.max_compression,
            p_tgt=list(cfg.train.compression.p_tgt),
            s_p=cfg.train.compression.s_p,
            concentration_control=cfg.train.compression.concentration_control,
            skip_prob=cfg.train.compression.skip_prob,
            use_paper_d_enforce=cfg.train.compression.get("use_paper_d_enforce", None),
        )
        return mm, None
    if kind == "cool":
        cm = CoolManager(
            down_sample_ratio=cfg.train.compression.down_sample_ratio,
            max_compression=cfg.train.compression.max_compression,
            n_jobs=cfg.train.compression.get("n_jobs", 1),
        )
        return None, cm
    raise ValueError(f"Unknown compression.kind: {kind!r}")


@hydra.main(config_path="../../external/BigCodec/config", config_name="default", version_base=None)
def train(cfg):
    callbacks = [_build_checkpoint_callback(cfg), LearningRateMonitor(logging_interval="step")]

    datamodule = DataModule(cfg)
    mm, cm = _build_compression(cfg)

    resume_path = cfg.get("ckpt", None)
    if resume_path:
        print(f"[train_melt] resuming Melt run from {resume_path}")
        lightning_module = CoolMeltWrapper(cfg, melt_manager=mm, cool_manager=cm)
        ckpt_path = resume_path
    else:
        backbone_ckpt = cfg.backbone_ckpt
        print(f"[train_melt] starting fresh Melt run from backbone {backbone_ckpt}")
        lightning_module = CoolMeltWrapper.load_from_checkpoint(
            backbone_ckpt,
            cfg=cfg,
            melt_manager=mm,
            cool_manager=cm,
            strict=True,
        )
        ckpt_path = None

    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=_build_logger(cfg),
        limit_train_batches=1.0 if not cfg.debug else 0.001,
    )
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"Melt training ends, ckpt path: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    train()
