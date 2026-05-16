#!/bin/bash
# 100-step single-GPU sanity run. Validates the full plumbing.
set -euo pipefail

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derive REPO root (assuming depth of 2: backbones/scripts/ or backbones/slurm/)
REPO="$(cd "$SCRIPT_DIR/../../" && pwd)"

# VENV should be activated by the user or set via environment variable
VENV="${VENV:-$REPO/.venv}"
if [ -d "$VENV" ]; then
    export PATH="$VENV/bin:$PATH"
    export VIRTUAL_ENV="$VENV"
fi

BIGCODEC=$REPO/external/BigCodec
CONFIG_NAME=${1:-codecslime_vq8k}
RUN_NAME=${2:-smoke-${CONFIG_NAME}}

# VENV should be activated by the user
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
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    train.trainer.devices=1 \
    train.trainer.max_steps=200 \
    train.trainer.min_steps=200 \
    train.checkpoint.every_n_train_steps=200 \
    train.logger.name=$RUN_NAME \
    train.logger.id=$RUN_NAME \
    log_dir=$REPO/backbones/checkpoints/smoke
