# CodecSlime

This repository contains a re-implementation of **CodecSlime** (Wang et al., arXiv 2506.21074), a plugin-style dynamic frame rate (DFR) wrapper for speech codecs. This project was developed as a TAU course final project.

CodecSlime uses a two-stage post-training pipeline (**Melt-and-Cool**) on top of a fixed-rate backbone (BigCodec) to achieve high-quality speech compression at variable bitrates. It consists of two primary innovations:
- **ScheDFR** (`sched_dfr.py`): A DP-based dynamic frame rate inference scheduler.
- **Melt-and-Cool** (`melt_manager.py`, `cool_manager.py`): A two-stage training recipe for adapting FFR models to DFR.

## Repository Structure

```text
backbones/
  scripts/        Python entry points (train / eval / data prep)
  configs/        Hydra YAMLs (model + train + entry points)
  slurm/          SLURM launchers
  checkpoints/    Trained model weights (gitignored)
  data/           Manifests: librispeech_*.txt, unicats_b.txt
  results/        Reconstructed audio + metrics per evaluation run
datasets/         Directory for LibriSpeech and LibriTTS datasets (gitignored)
external/BigCodec Vendored BigCodec backbone (gitignored, commit pinned)
docs/             Additional documentation and training plans
papers/           The original CodecSlime paper PDF + LaTeX source
```

## Setup Instructions

### 1. Setup BigCodec Backbone

The BigCodec backbone is vendored in `external/BigCodec`. To populate the remaining backbone files from the upstream repository (using the commit pinned in `external/BIGCODEC_COMMIT.txt`) without overwriting existing project-specific modifications:

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

### 2. Environment Setup

Create a virtual environment and install the required dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Note: For AMD/ROCm users (e.g., node n-210), refer to `docs/n210_amd_setup.md` for specific environment instructions.*

### 3. Data Preparation

Download and prepare the LibriSpeech (for training) and LibriTTS (for evaluation) datasets.

```bash
# Download and prepare datasets
python backbones/scripts/prepare_librispeech.py --root datasets
python backbones/scripts/prepare_libritts.py --root datasets --subsets test-clean

# Build the UniCATS testset B manifest (500 utts, 37 unseen speakers)
python backbones/scripts/prepare_unicats_b.py --out-dir backbones/data
```

## Reproduction Workflow

Note currently the account in the slurm files uses a placeholder:
```
--account=<your_account>
```
Please update it to the currect one

### Stage 1: Backbone Training (FFR)

Train the BigCodec backbone at a fixed 80 Hz frame rate.

```bash
sbatch backbones/slurm/train_vq8k.slurm
sbatch backbones/slurm/train_fsq18k.slurm
```

### Stage 2: Melt Post-Training

Apply the random-rate downsampling curriculum (`MeltManager`) to train the model to tolerate arbitrary segment lengths.

```bash
# L40s cluster, batch 16, 100k steps (paper-literal LR)
sbatch backbones/slurm/train_melt_vq8k.slurm
sbatch backbones/slurm/train_melt_fsq18k.slurm

# AMD n-210 cluster, batch 64, 12.5k steps (sqrt-scaled LR)
sbatch backbones/slurm/train_melt_vq8k_n210.slurm
sbatch backbones/slurm/train_melt_fsq18k_n210.slurm
```

### Stage 3: Cool Fine-Tuning

Fine-tune the quantizer and decoder with ScheDFR enabled while freezing the encoder.

```bash
# Standard pipeline (Full Melt+Cool)
sbatch backbones/slurm/train_cool_vq8k_n210.slurm
sbatch backbones/slurm/train_cool_fsq18k_n210.slurm

# Ablation: Cool from backbone only (no Melt)
sbatch backbones/slurm/train_cool_vq8k.slurm
```

### Stage 4: Evaluation

Evaluate models using the `evaluate_codec.py` script. It computes WER, STOI, PESQ, SECS, and UTMOS.

```bash
python backbones/scripts/evaluate_codec.py \
    --ckpt backbones/checkpoints/cool-fsq18k-n210-12500/last.ckpt \
    --manifest backbones/data/unicats_b.txt \
    --audio-root datasets/LibriTTS \
    --mode dfr --rs 2.0 --u 4 \
    --out-dir backbones/results/eval-meltcool-fsq18k-n210 \
    --codebook-size 18225 --whisper-model base
```

Alternatively, run the full 12-cell evaluation matrix via SLURM:
```bash
sbatch backbones/slurm/eval_codec.slurm
```

## Evaluation Matrix & Metrics

### Matrix Variants

| Variant | Mode | Codebook | Target Bitrate |
|---|---|---|---|
| backbone-vq8k-ffr | ffr | 8192 | 1040 bps |
| backbone-vq8k-dfr | dfr | 8192 | 600 bps |
| melt-vq8k-n210 | dfr | 8192 | 600 bps |
| meltcool-vq8k-n210 | dfr | 8192 | 600 bps |
| backbone-fsq18k-ffr | ffr | 18225 | 1132 bps |
| meltcool-fsq18k-n210 | dfr | 18225 | 646 bps |

*Refer to `backbones/results/final/all_metrics.md` for the full results.*

**Bitrate Formula:** `(log2(codebook) + ceil(log2(U))) * encoder_fr / mean_comp_ratio`

### Metrics Details

| Metric | Direction | Backend | Notes |
|---|---|---|---|
| Bitrate | n/a | Closed form | Includes duration bits for DFR |
| WER | lower better | OpenAI Whisper (`base`) | Transcription accuracy |
| STOI | higher better | pystoi | Intelligibility |
| PESQ | higher better | pesq (wb) | Perceptual quality (16 kHz) |
| SECS | higher better | Resemblyzer | Speaker similarity |
| UTMOS | higher better | SpeechMOS | Neural MOS predictor |


## Audio Samples

You can listen to audio samples demonstrating the compression and reconstruction quality across different model variants in the `backbones/results/samples/` directory. 

For example, the samples for the FSQ-18K dynamic frame rate variant can be found here:
[`backbones/results/samples/coolmelt-fsq18k-n210-dfr40`](https://github.com/NacOJerk/audio_ml_tau_final/tree/main/backbones/results/samples/coolmelt-fsq18k-n210-dfr40)

Each sample set includes:
* `*_orig.wav`: The original, uncompressed source audio.
* `*_recon.wav`: The reconstructed audio after passing through the Melt-and-Cool trained codec pipeline.
* `*_ref.txt`: The corresponding reference transcription text.

## Contributing

This project was developed as part of a TAU course final project. For more details on implementation decisions, see the documentation in the `docs/` folder.
