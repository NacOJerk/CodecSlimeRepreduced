#!/bin/bash
# 100-step single-GPU sanity run. Validates the full plumbing.
set -euo pipefail

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
CONFIG_NAME=${1:-codecslime_vq8k}
RUN_NAME=${2:-smoke-${CONFIG_NAME}}

VENV=/home/morg/students/dortirosh/envs/codecslime
export PATH=$VENV/bin:${PATH:-}
export VIRTUAL_ENV=$VENV
export PYTHONPATH=$BIGCODEC:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Build filelists if not yet built (smoke needs at least a few hundred utterances)
if [ ! -f "$REPO/backbones/data/librispeech_train.txt" ]; then
  mkdir -p "$REPO/backbones/data"
  cd "$BIGCODEC"
  python preprocess.py \
    hydra.output_subdir=null hydra.job.chdir=False \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    preprocess.view.train_filelist=$REPO/backbones/data/librispeech_train.txt \
    preprocess.view.test_filelist=$REPO/backbones/data/librispeech_test.txt
fi

cd "$BIGCODEC"
python train.py \
    --config-dir=$REPO/backbones/configs \
    --config-name=$CONFIG_NAME \
    train.trainer.devices=1 \
    train.trainer.max_steps=200 \
    train.trainer.min_steps=200 \
    train.checkpoint.every_n_train_steps=200 \
    train.logger.name=$RUN_NAME \
    train.logger.id=$RUN_NAME \
    log_dir=$REPO/backbones/checkpoints/smoke
