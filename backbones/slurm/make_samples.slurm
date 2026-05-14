#!/bin/bash
#SBATCH --job-name=cs-samples
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%A_%a.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-13

# Generate the first N=5 orig/recon utterance pairs from UniCATS-B for every
# evaluated configuration, into backbones/results/samples/<variant>/. No
# Whisper, no UTMOS - we only need the audio for manual listening. STOI/PESQ
# /SECS are still cheap and stay on so the metrics_summary.json captures a
# quick objective snapshot per sample.

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
PYBIN=/home/morg/students/dortirosh/envs/codecslime/bin/python
N_SAMPLES=5

VARIANTS=(
  "backbone-vq8k-ffr80                  vq8k-300k                            ffr    8192"
  "backbone-vq8k-ffr40                  vq8k-300k                            fixed  8192"
  "backbone-vq8k-dfr40                  vq8k-300k                            dfr    8192"
  "backbone-fsq18k-ffr80                fsq18k-300k                          ffr    18225"
  "backbone-fsq18k-ffr40                fsq18k-300k                          fixed  18225"
  "backbone-fsq18k-dfr40                fsq18k-300k                          dfr    18225"
  "melt-fsq18k-n210-ffr40               melt-fsq18k-n210-12500               fixed  18225"
  "melt-fsq18k-n210-dfr40               melt-fsq18k-n210-12500               dfr    18225"
  "melt-vq8k-n210-paperd-ffr40          melt-vq8k-n210-paperd-12500          fixed  8192"
  "melt-vq8k-n210-paperd-dfr40          melt-vq8k-n210-paperd-12500          dfr    8192"
  "cool-vq8k-from-backbone-n210-dfr40   cool-vq8k-from-backbone-n210-12500   dfr    8192"
  "coolmelt-vq8k-n210-dfr40             cool-vq8k-n210-12500                 dfr    8192"
  "cool-fsq18k-from-backbone-n210-dfr40 cool-fsq18k-from-backbone-n210-12500 dfr    18225"
  "coolmelt-fsq18k-n210-dfr40           cool-fsq18k-n210-12500               dfr    18225"
)

ROW="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
read VARIANT CKPT_DIR MODE CODEBOOK <<< "$ROW"
CKPT=$REPO/backbones/checkpoints/$CKPT_DIR/last.ckpt
OUT=$REPO/backbones/results/samples/$VARIANT

if [ ! -f "$CKPT" ]; then
    echo "[skip] $CKPT not present; nothing to sample for $VARIANT"
    exit 0
fi

mkdir -p "$REPO/backbones/slurm/logs"
echo "[samples] node=$(hostname) variant=$VARIANT mode=$MODE codebook=$CODEBOOK ckpt=$CKPT out=$OUT n=$N_SAMPLES"

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
    --limit $N_SAMPLES \
    --no-wer \
    --no-utmos
