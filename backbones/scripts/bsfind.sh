#!/bin/bash
# Single training-step memory probe at a given batch size.
# Args: $1 = batch size, $2 = run name (defaults to bsfind-bs$1-<ts>)
set -euo pipefail

BS=${1:?need batch size}
RUN=${2:-bsfind-bs$BS-$(date +%s)}

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
# VENV should be activated by the user or set via environment variable
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
