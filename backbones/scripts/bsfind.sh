#!/bin/bash
# Single training-step memory probe at a given batch size.
# Args: $1 = batch size, $2 = run name (defaults to bsfind-bs$1-<ts>)
set -euo pipefail

BS=${1:?need batch size}
RUN=${2:-bsfind-bs$BS-$(date +%s)}

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
VENV=/home/morg/students/dortirosh/envs/codecslime

export PATH=$VENV/bin:${PATH:-}
export VIRTUAL_ENV=$VENV
export PYTHONPATH=$BIGCODEC:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== host=$(hostname) bs=$BS run=$RUN ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $BIGCODEC
python train.py \
    --config-dir=$REPO/backbones/configs \
    --config-name=codecslime_vq8k \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    train.trainer.devices=1 \
    train.trainer.max_steps=50 train.trainer.min_steps=50 \
    train.checkpoint.every_n_train_steps=50 \
    dataset.train.batch_size=$BS \
    train.logger.name=$RUN train.logger.id=$RUN \
    log_dir=$REPO/backbones/checkpoints/$RUN

echo "=== bs=$BS PASSED ==="
