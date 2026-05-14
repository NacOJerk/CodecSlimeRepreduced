#!/bin/bash
# Melt-stage launcher for the AMD/ROCm (n-210) environment.
# Args:
#   $1 = config name (e.g. codecslime_melt_vq8k_n210)
#   $2 = run name / W&B id (e.g. melt-vq8k-n210-12500)
set -euo pipefail

CFG=$1
shift
RUN=$1
shift
EXTRA_ARGS=("$@")

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
LOG_DIR=$REPO/backbones/checkpoints/$RUN

VENV=/home/morg/students/dortirosh/envs/cs_amd
export PATH=$VENV/bin:${PATH:-}
export VIRTUAL_ENV=$VENV
export PYTHONPATH=$REPO:$BIGCODEC:${PYTHONPATH:-}

# ROCm hint: turn off MIOpen find-mode caching the first time we touch it.
# PYTORCH_CUDA_ALLOC_CONF is CUDA-only, harmless on ROCm but skip it to be tidy.
export MIOPEN_FIND_MODE=NORMAL

mkdir -p "$LOG_DIR"

if [ ! -f "$REPO/backbones/data/librispeech_train.txt" ]; then
  mkdir -p "$REPO/backbones/data"
  cd "$BIGCODEC"
  python preprocess.py \
    hydra.output_subdir=null hydra.job.chdir=False \
    preprocess.datasets.LibriSpeech.root=$REPO/datasets/LibriSpeech \
    preprocess.view.train_filelist=$REPO/backbones/data/librispeech_train.txt \
    preprocess.view.test_filelist=$REPO/backbones/data/librispeech_test.txt
fi

RESUME_ARG=()
if [ -f "$LOG_DIR/last.ckpt" ]; then
  RESUME_ARG=("ckpt=$LOG_DIR/last.ckpt")
  echo "[auto_resume_melt_amd] resuming from $LOG_DIR/last.ckpt"
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
