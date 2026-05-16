#!/bin/bash
# Resume-aware launcher. Args:
#   $1 = config name (codecslime_vq8k | codecslime_fsq18k)
#   $2 = run name / W&B id (e.g. vq8k-300k)
set -euo pipefail

CFG=$1
RUN=$2

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
LOG_DIR=$REPO/backbones/checkpoints/$RUN

# VENV should be activated by the user
# export PYTHONPATH=$BIGCODEC:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$LOG_DIR"

# Build filelists if not present
if [ ! -f "$REPO/backbones/data/librispeech_train.txt" ]; then
  mkdir -p "$REPO/backbones/data"
  cd "$BIGCODEC"
  python preprocess.py \
    hydra.output_subdir=null hydra.job.chdir=False \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    preprocess.view.train_filelist=$REPO/backbones/data/librispeech_train.txt \
    preprocess.view.test_filelist=$REPO/backbones/data/librispeech_test.txt
fi

# Find last checkpoint
RESUME_ARG=()
if [ -f "$LOG_DIR/last.ckpt" ]; then
  RESUME_ARG=("ckpt=$LOG_DIR/last.ckpt")
  echo "[auto_resume] resuming from $LOG_DIR/last.ckpt"
fi

cd "$BIGCODEC"
python train.py \
    --config-dir="$REPO/backbones/configs" \
    --config-name="$CFG" \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    train.logger.name="$RUN" \
    train.logger.id="$RUN" \
    log_dir="$LOG_DIR" \
    "${RESUME_ARG[@]}"
