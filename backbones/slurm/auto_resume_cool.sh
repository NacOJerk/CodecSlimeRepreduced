#!/bin/bash
# Cool-stage launcher (L40s / CUDA env). Args:
#   $1 = config name (codecslime_cool_vq8k | codecslime_cool_fsq18k | ...)
#   $2 = run name / W&B id (e.g. cool-vq8k-from-backbone-100k)
set -euo pipefail

CFG=$1
shift
RUN=$1
shift
EXTRA_ARGS=("$@")

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
# VENV=<VENV_PATH>/codecslime
# export PATH=$VENV/bin:${PATH:-}
# export VIRTUAL_ENV=$VENV
export PYTHONPATH=$REPO:$BIGCODEC:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Keep BLAS thread count at 1 per process so joblib's per-item workers do not
# oversubscribe the slurm cpus-per-task allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$LOG_DIR"

# Build filelists if not present (same as backbone setup)
if [ ! -f "$REPO/backbones/data/librispeech_train.txt" ]; then
  mkdir -p "$REPO/backbones/data"
  cd "$BIGCODEC"
  python preprocess.py \
    hydra.output_subdir=null hydra.job.chdir=False \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    preprocess.view.train_filelist=$REPO/backbones/data/librispeech_train.txt \
    preprocess.view.test_filelist=$REPO/backbones/data/librispeech_test.txt
fi

# Resume from in-progress Cool checkpoint if present (independent of backbone_ckpt).
RESUME_ARG=()
if [ -f "$LOG_DIR/last.ckpt" ]; then
  RESUME_ARG=("ckpt=$LOG_DIR/last.ckpt")
  echo "[auto_resume_cool] resuming from $LOG_DIR/last.ckpt"
fi

cd "$REPO"
python backbones/scripts/train_melt.py \
    --config-dir="$REPO/backbones/configs" \
    --config-name="$CFG" \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    train.logger.name="$RUN" \
    train.logger.id="$RUN" \
    log_dir="$LOG_DIR" \
    "${EXTRA_ARGS[@]}" \
    "${RESUME_ARG[@]}"
