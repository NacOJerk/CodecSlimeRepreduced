#!/bin/bash
#SBATCH --job-name=cs-eval-codec
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-11

# Evaluates the full backbone x stage x mode matrix on UniCATS testset B.
# Each task index runs one (ckpt, mode) cell. Missing checkpoints are skipped
# with exit 0 so the array tolerates Melt / Cool runs that have not finished.

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
PYBIN=/home/morg/students/dortirosh/envs/codecslime/bin/python

VARIANTS=(
  "backbone-vq8k-ffr      vq8k-300k                        ffr  8192"
  "backbone-vq8k-dfr      vq8k-300k                        dfr  8192"
  "melt-vq8k-l40s         melt-vq8k-100k                   dfr  8192"
  "melt-vq8k-n210         melt-vq8k-n210-12500             dfr  8192"
  "cool-vq8k-l40s         cool-vq8k-from-backbone-100k     dfr  8192"
  "meltcool-vq8k-n210     cool-vq8k-n210-12500             dfr  8192"
  "backbone-fsq18k-ffr    fsq18k-300k                      ffr  18225"
  "backbone-fsq18k-dfr    fsq18k-300k                      dfr  18225"
  "melt-fsq18k-l40s       melt-fsq18k-100k                 dfr  18225"
  "melt-fsq18k-n210       melt-fsq18k-n210-12500           dfr  18225"
  "cool-fsq18k-l40s       cool-fsq18k-from-backbone-100k   dfr  18225"
  "meltcool-fsq18k-n210   cool-fsq18k-n210-12500           dfr  18225"
)

ROW="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
read VARIANT CKPT_DIR MODE CODEBOOK <<< "$ROW"
CKPT=$REPO/backbones/checkpoints/$CKPT_DIR/last.ckpt
OUT=$REPO/backbones/results/eval-$VARIANT

if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not present; nothing to eval for $VARIANT"
    exit 0
fi

mkdir -p "$REPO/backbones/slurm/logs"
echo "[eval] variant=$VARIANT mode=$MODE codebook=$CODEBOOK ckpt=$CKPT"
echo "[eval] out=$OUT"

export PYTHONPATH=$REPO:$REPO/external/BigCodec:${PYTHONPATH:-}

$PYBIN $REPO/backbones/scripts/evaluate_codec.py \
    --ckpt "$CKPT" \
    --manifest $REPO/backbones/data/unicats_b.txt \
    --audio-root $REPO/datasets/LibriTTS \
    --mode $MODE \
    --rs 2.0 \
    --u 4 \
    --out-dir "$OUT" \
    --codebook-size $CODEBOOK \
    --whisper-model base
