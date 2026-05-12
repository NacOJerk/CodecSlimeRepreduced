# audio_ml_tau_final

## Reproduction Workflow

All scripts are run from the **repo root**. Checkpoints go in `models/`, audio data in `data/`.

---

### Stage 1 — Vanilla Reconstruction

Encode + decode samples at the fixed codec frame rate (no compression).

```bash
# FSQ-18k checkpoint
python backbones/scripts/reconstruct_samples.py \
    --ckpt models/fsq18k-300k-inference.ckpt \
    --out-dir backbones/results/vanilla-fsq18k \
    --n-samples 10

# VQ-8k checkpoint
python backbones/scripts/reconstruct_samples.py \
    --ckpt models/vq8k-300k-inference.ckpt \
    --out-dir backbones/results/vanilla-vq8k \
    --n-samples 10
```

Outputs `{fid}_orig.wav` and `{fid}_recon.wav` into `--out-dir`, plus a `reconstruction_summary.txt` with per-sample SNR and average.

#### Vanilla Metrics Report

```bash
# FSQ-18k  (codebook size 18225)
python backbones/scripts/eval_metrics.py \
    --dir backbones/results/vanilla-fsq18k \
    --codebook-size 18225 \
    --frame-rate 80 \
    --out backbones/results/vanilla-fsq18k/metrics.txt

# VQ-8k  (codebook size 8192)
python backbones/scripts/eval_metrics.py \
    --dir backbones/results/vanilla-vq8k \
    --codebook-size 8192 \
    --frame-rate 80 \
    --out backbones/results/vanilla-vq8k/metrics.txt
```

---

### Stage 2 — COOL (DFR Compression at Inference)

Apply Dynamic Frame Rate compression without retraining. Produces both a full reconstruction and a COOL-compressed reconstruction for every sample.

```bash
# FSQ-18k, compression ratio ~2×
python backbones/scripts/evaluate_cool_simple.py \
    --ckpt models/fsq18k-300k-inference.ckpt \
    --out-dir backbones/results/cool-fsq18k \
    --down-sample-ratio 2.0 \
    --max-compression 4 \
    --n-samples 10

# VQ-8k, compression ratio ~2×
python backbones/scripts/evaluate_cool_simple.py \
    --ckpt models/vq8k-300k-inference.ckpt \
    --out-dir backbones/results/cool-vq8k \
    --down-sample-ratio 2.0 \
    --max-compression 4 \
    --n-samples 10
```

Outputs `{fid}_orig.wav`, `{fid}_full_recon.wav`, and `{fid}_cool_recon.wav` per sample, plus `cool_evaluation_summary.txt` with SNR and compression ratio stats.

#### COOL Metrics Report

The `--recon-suffix` selects which reconstructions to evaluate. Use `_full_recon` for the uncompressed baseline and `_cool_recon` for the DFR-compressed output.

```bash
# FSQ-18k — COOL reconstructions (compression ratio ~2, so bitrate halved)
python backbones/scripts/eval_metrics.py \
    --dir backbones/results/cool-fsq18k \
    --orig-suffix _orig \
    --recon-suffix _cool_recon \
    --codebook-size 18225 \
    --frame-rate 80 \
    --compression-ratio 2.0 \
    --out backbones/results/cool-fsq18k/metrics_cool.txt

# FSQ-18k — full (uncompressed) reconstructions from the same run
python backbones/scripts/eval_metrics.py \
    --dir backbones/results/cool-fsq18k \
    --orig-suffix _orig \
    --recon-suffix _full_recon \
    --codebook-size 18225 \
    --frame-rate 80 \
    --out backbones/results/cool-fsq18k/metrics_full.txt
```

---

### Metrics Reported

| Metric | Direction | Notes |
|--------|-----------|-------|
| Bitrate | — | `log2(codebook_size) × frame_rate / compression_ratio` bps |
| WER | ↓ lower is better | Whisper transcription error rate |
| STOI | ↑ | Short-time objective intelligibility |
| PESQ | ↑ | Perceptual speech quality |
| SECS | ↑ | Speaker embedding cosine similarity |
| UTMOS | ↑ | Neural MOS predictor |

Install metric dependencies:

```bash
pip install openai-whisper jiwer pystoi pesq resemblyzer
```