#!/bin/bash
# Melt-stage launcher. Args:
#   $1 = config name (codecslime_melt_vq8k | codecslime_melt_fsq18k)
#   $2 = run name / W&B id (e.g. melt-vq8k-100k)
set -euo pipefail

CFG=$1
shift
RUN=$1
shift
EXTRA_ARGS=("$@")

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
LOG_DIR=$REPO/backbones/checkpoints/$RUN

VENV=/home/morg/students/dortirosh/envs/codecslime
export PATH=$VENV/bin:${PATH:-}
export VIRTUAL_ENV=$VENV
export PYTHONPATH=$REPO:$BIGCODEC:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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

# Resume from in-progress Melt checkpoint if present (independent of backbone_ckpt).
RESUME_ARG=()
if [ -f "$LOG_DIR/last.ckpt" ]; then
  RESUME_ARG=("ckpt=$LOG_DIR/last.ckpt")
  echo "[auto_resume_melt] resuming from $LOG_DIR/last.ckpt"
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
