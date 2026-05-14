#!/bin/bash
#SBATCH --job-name=cs-eval-l40s
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11

# Sister of eval_codec_n102.slurm targeting the killable L40s pool instead of
# n-102. Identical 12-cell VARIANTS list; submit via `sbatch --array=...` with
# whichever indices you want to land on L40s. Used as a relief valve when
# n-102 is contended.

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
PYBIN=/home/morg/students/dortirosh/envs/codecslime/bin/python

VARIANTS=(
  "backbone-vq8k-ffr80      vq8k-300k                ffr    8192"
  "backbone-vq8k-ffr40      vq8k-300k                fixed  8192"
  "backbone-vq8k-dfr40      vq8k-300k                dfr    8192"
  "backbone-fsq18k-ffr80    fsq18k-300k              ffr    18225"
  "backbone-fsq18k-ffr40    fsq18k-300k              fixed  18225"
  "backbone-fsq18k-dfr40    fsq18k-300k              dfr    18225"
  # melt-vq8k-n210-12500 stopped at step 7500/12500 (not finished); excluded.
  "DISABLED-not-finished    melt-vq8k-n210-12500     ffr    8192"
  "DISABLED-not-finished    melt-vq8k-n210-12500     ffr    8192"
  "melt-fsq18k-n210-ffr40   melt-fsq18k-n210-12500   fixed  18225"
  "melt-fsq18k-n210-dfr40   melt-fsq18k-n210-12500   dfr    18225"
  "melt-vq8k-n210-paperd-ffr40  melt-vq8k-n210-paperd-12500  fixed  8192"
  "melt-vq8k-n210-paperd-dfr40  melt-vq8k-n210-paperd-12500  dfr    8192"
)

ROW="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
read VARIANT CKPT_DIR MODE CODEBOOK <<< "$ROW"
CKPT=$REPO/backbones/checkpoints/$CKPT_DIR/last.ckpt
OUT=$REPO/backbones/results/eval-$VARIANT

if [[ "$VARIANT" == DISABLED* ]]; then
    echo "[skip] $VARIANT marked DISABLED (model not finished)"
    exit 0
fi
if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not present; nothing to eval for $VARIANT"
    exit 0
fi

mkdir -p "$REPO/backbones/slurm/logs"
echo "[eval] node=$(hostname) variant=$VARIANT mode=$MODE codebook=$CODEBOOK"
echo "[eval] ckpt=$CKPT"
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
