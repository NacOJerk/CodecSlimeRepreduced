# CodecSlime

This repository contains a re-implementation of **CodecSlime** (Wang et al., arXiv 2506.21074), a plugin-style dynamic frame rate (DFR) wrapper for speech codecs. 

CodecSlime uses a two-stage post-training pipeline (**Melt-and-Cool**) on top of a fixed-rate backbone (BigCodec) to achieve high-quality speech compression at variable bitrates.

## Repository Structure

- `backbones/`: Configuration, scripts, and results for the codec models.
- `datasets/`: Directory for LibriSpeech and LibriTTS datasets (gitignored).
- `external/`: External dependencies, including the BigCodec backbone.
- `docs/`: Additional documentation and training plans.
- `papers/`: The original CodecSlime paper.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd audio_ml_tau_final
```

### 2. Setup BigCodec Backbone

The BigCodec backbone is vendored in `external/BigCodec`. This directory already exists and contains project-specific modifications. To populate the remaining backbone files from the upstream repository (using the commit pinned in `external/BIGCODEC_COMMIT.txt`) without overwriting existing files:

```bash
# Clone upstream to a temporary location
git clone https://github.com/m-m-y/BigCodec.git external/BigCodec_upstream
cd external/BigCodec_upstream
git checkout $(cat ../BIGCODEC_COMMIT.txt)
cd ../..

# Merge upstream files into local directory (no-clobber)
cp -rn external/BigCodec_upstream/* external/BigCodec/
rm -rf external/BigCodec_upstream
```

### 3. Environment Setup

Create a virtual environment and install the required dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Note: For AMD/ROCm users, refer to `docs/n210_amd_setup.md` for specific environment instructions.*

### 4. Data Preparation

Download and prepare the LibriSpeech (for training) and LibriTTS (for evaluation) datasets.

```bash
# Download and prepare datasets
python backbones/scripts/prepare_librispeech.py --root datasets
python backbones/scripts/prepare_libritts.py --root datasets --subsets test-clean

# Build the UniCATS testset B manifest
python backbones/scripts/prepare_unicats_b.py --out-dir backbones/data
```

## Reproduction Workflow

### Stage 1: Backbone Training (FFR)

Train the BigCodec backbone at a fixed 80 Hz frame rate.

```bash
sbatch backbones/slurm/train_vq8k.slurm
sbatch backbones/slurm/train_fsq18k.slurm
```

### Stage 2: Melt Post-Training

Apply the random-rate downsampling curriculum to train the model to tolerate arbitrary segment lengths.

```bash
sbatch backbones/slurm/train_melt_vq8k.slurm
sbatch backbones/slurm/train_melt_fsq18k.slurm
```

### Stage 3: Cool Fine-Tuning

Fine-tune the quantizer and decoder with ScheDFR enabled while freezing the encoder.

```bash
sbatch backbones/slurm/train_cool_vq8k_n210.slurm
sbatch backbones/slurm/train_cool_fsq18k_n210.slurm
```

### Stage 4: Evaluation

Evaluate the resulting model using the `evaluate_codec.py` script. This script computes WER, STOI, PESQ, SECS, and UTMOS.

```bash
python backbones/scripts/evaluate_codec.py \
    --ckpt backbones/checkpoints/cool-fsq18k-n210-12500/last.ckpt \
    --manifest backbones/data/unicats_b.txt \
    --audio-root datasets/LibriTTS \
    --mode dfr --rs 2.0 --u 4 \
    --out-dir backbones/results/eval-meltcool-fsq18k-n210 \
    --codebook-size 18225 --whisper-model base
```

## Evaluation Matrix

The following variants are evaluated in the paper and this re-implementation:

| Variant | Mode | Codebook | Target Bitrate |
|---|---|---|---|
| Backbone VQ-8k | FFR | 8192 | 1040 bps |
| Backbone VQ-8k | DFR | 8192 | 600 bps |
| Melt VQ-8k | DFR | 8192 | 600 bps |
| CodecSlime (Melt+Cool) | DFR | 8192 | 600 bps |

Refer to the main `README.md` (original) or `backbones/results/final/all_metrics.md` for detailed results.

## Contributing

This project was developed as part of a TAU course final project. For more details on the implementation decisions, see the documentation in the `docs/` folder.
