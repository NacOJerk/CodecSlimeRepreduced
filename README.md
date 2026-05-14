# audio_ml_tau_final

TAU course final project: from-scratch re-implementation of **CodecSlime**
(Wang et al., arXiv 2506.21074). Plugin-style dynamic frame rate wrapper on a
BigCodec VQ-GAN speech codec, with two components:

- **ScheDFR** (`sched_dfr.py`): DP-based inference scheduler.
- **Melt-and-Cool** (`melt_manager.py`, `cool_manager.py`): two-stage post-training.

Paper PDF lives in `papers/codecslime_2506.21074.pdf`.

## Repository layout

```
backbones/
  scripts/        Python entry points (train / eval / data prep)
  configs/        Hydra YAMLs (model + train + entry points)
  slurm/          SLURM launchers
  checkpoints/    Trained model weights (gitignored, ~14 TB)
  data/           Manifests: librispeech_*.txt, unicats_b.txt
  results/        Reconstructed audio + metrics per evaluation run
datasets/         LibriSpeech, LibriTTS (gitignored)
external/BigCodec Vendored backbone (gitignored, commit pinned)
papers/           Paper PDF + LaTeX source
```

## Setup

```bash
# Activate the conda env (CUDA path; n-210 AMD uses a separate cs_amd env)
source /home/morg/students/dortirosh/envs/codecslime/bin/activate

# Install eval metric backends (if missing)
pip install -r requirements.txt
```

## Reproduction workflow

All commands run from the repo root.

### Stage 1: Backbone FFR training

Train the BigCodec backbone at fixed 80 Hz frame rate (paper Section 3.1).
Outputs land in `backbones/checkpoints/{vq8k,fsq18k}-300k/`.

```bash
sbatch backbones/slurm/train_vq8k.slurm
sbatch backbones/slurm/train_fsq18k.slurm
```

### Stage 2: Melt post-training (Section 2.3 of paper)

Random-rate downsampling curriculum (`MeltManager`). Pretrains the codec to
tolerate arbitrary segment lengths, producing a DFR-foundation model.

```bash
# L40s killable, batch 16, 100k steps (paper-literal LR)
sbatch backbones/slurm/train_melt_vq8k.slurm
sbatch backbones/slurm/train_melt_fsq18k.slurm

# AMD n-210, batch 64, 12.5k steps (sqrt-scaled LR)
sbatch backbones/slurm/train_melt_vq8k_n210.slurm
sbatch backbones/slurm/train_melt_fsq18k_n210.slurm
```

### Stage 3: Cool fine-tuning (Section 2.3 of paper)

Encoder frozen; quantizer + decoder fine-tuned with ScheDFR enabled per step
(`CoolManager`). Two pipelines:

- Cool from backbone (paper Table 3 ablation row "+Cool only"):
  ```bash
  sbatch backbones/slurm/train_cool_vq8k.slurm
  ```
  Output: `backbones/checkpoints/cool-vq8k-from-backbone-100k/`.

- Cool on top of Melt (the full Melt+Cool pipeline, paper Table 1 row
  "CodecSlime"):
  ```bash
  sbatch backbones/slurm/train_cool_vq8k_from_backbone_n210.slurm
  sbatch backbones/slurm/train_cool_vq8k_n210.slurm
  sbatch backbones/slurm/train_cool_fsq18k_n210.slurm
  ```
  Output: `backbones/checkpoints/cool-{vq8k,fsq18k}-n210-12500/`.

### Stage 4: Evaluation

Single end-to-end script: `backbones/scripts/evaluate_codec.py`. Generates the
reconstruction and computes WER, STOI, PESQ, SECS, UTMOS plus a bitrate
accounting, all in one invocation. Defaults match paper Table 1 (Rs=2, U=4).

```bash
# One cell, e.g. the full CodecSlime on FSQ-18k (Melt+Cool n210):
python backbones/scripts/evaluate_codec.py \
    --ckpt backbones/checkpoints/cool-fsq18k-n210-12500/last.ckpt \
    --manifest backbones/data/unicats_b.txt \
    --audio-root datasets/LibriTTS \
    --mode dfr --rs 2.0 --u 4 \
    --out-dir backbones/results/eval-meltcool-fsq18k-n210 \
    --codebook-size 18225 --whisper-model base
```

The slurm job array runs the full 12-cell matrix at once and skips missing
checkpoints (Melt or Cool runs still training):

```bash
sbatch backbones/slurm/eval_codec.slurm
```

Each cell writes `{fid}_orig.wav`, `{fid}_recon.wav`, optional `{fid}_ref.txt`
(LibriTTS `.normalized.txt` transcript), `metrics_summary.json`, and
`metrics.tsv` to `backbones/results/eval-<variant>/`.

#### Evaluation matrix

| Variant (slurm idx) | Checkpoint dir | Mode | Codebook | Bitrate at Rs=2 |
|---|---|---|---|---|
| backbone-vq8k-ffr (0) | vq8k-300k | ffr | 8192 | 1040 bps |
| backbone-vq8k-dfr (1) | vq8k-300k | dfr | 8192 | 600 bps |
| melt-vq8k-l40s (2) | melt-vq8k-100k | dfr | 8192 | 600 bps |
| melt-vq8k-n210 (3) | melt-vq8k-n210-12500 | dfr | 8192 | 600 bps |
| cool-vq8k-l40s (4) | cool-vq8k-from-backbone-100k | dfr | 8192 | 600 bps |
| meltcool-vq8k-n210 (5) | cool-vq8k-n210-12500 | dfr | 8192 | 600 bps |
| backbone-fsq18k-ffr (6) | fsq18k-300k | ffr | 18225 | 1132 bps |
| backbone-fsq18k-dfr (7) | fsq18k-300k | dfr | 18225 | 646 bps |
| melt-fsq18k-l40s (8) | melt-fsq18k-100k | dfr | 18225 | 646 bps |
| melt-fsq18k-n210 (9) | melt-fsq18k-n210-12500 | dfr | 18225 | 646 bps |
| cool-fsq18k-l40s (10) | cool-fsq18k-from-backbone-100k | dfr | 18225 | 646 bps |
| meltcool-fsq18k-n210 (11) | cool-fsq18k-n210-12500 | dfr | 18225 | 646 bps |

Bitrate formula:
`(log2(codebook) + ceil(log2(U))) * encoder_fr / mean_comp_ratio`, where the
duration bits are 0 for FFR (`ceil(log2(U))=0` when U=1) and 2 for DFR with
U=4. `mean_comp_ratio` averages near Rs=2 on real audio.

### Data preparation

LibriSpeech (training) and LibriTTS test-clean (evaluation) are downloaded
by helper scripts:

```bash
python backbones/scripts/prepare_librispeech.py --root datasets
python backbones/scripts/prepare_libritts.py --root datasets --subsets test-clean

# Build the UniCATS testset B manifest (500 utts, 37 unseen speakers)
python backbones/scripts/prepare_unicats_b.py --out-dir backbones/data
```

The UniCATS-B script writes `backbones/data/unicats_b.txt` (manifest used by
`evaluate_codec.py`) and `backbones/data/unicats_b_utt2prompt.txt` (verbatim
upstream pair list, kept for traceability). Manifests live under
`backbones/data/` which is gitignored; regenerate via the scripts above.

## Metrics reported

| Metric | Direction | Backend | Notes |
|---|---|---|---|
| Bitrate | n/a | closed form | log2(codebook) + duration bits times encoder_fr / mean_comp_ratio |
| WER | lower better | OpenAI Whisper (`base` default) | Paper uses NeMo FastConformer; absolute numbers will differ |
| STOI | higher better | pystoi | Short-time objective intelligibility |
| PESQ | higher better | pesq (mode='wb') | Perceptual speech quality at 16 kHz |
| SECS | higher better | Resemblyzer (paper-faithful) | Speaker embedding cosine similarity |
| UTMOS | higher better | tarepan/SpeechMOS via torch.hub | Neural MOS predictor |

ViSQOL is not computed (no install path); the STOI + PESQ + UTMOS combination
covers the same intelligibility / perceptual dimensions. WER uses Whisper
instead of the paper's NeMo FastConformer-Transducer-Large; rank within the
matrix is comparable but absolute Table 1 numbers will not match.

## SLURM cluster notes

| Node class | Partition | Time | GPU type |
|---|---|---|---|
| t-100 (H100 80 GB) | gpu-morgeva | 5 days | gpu:h100:1 |
| n-801..805 (L40s 48 GB) | killable | 1 day | gpu:l40s:1 |
| n-210 (AMD MI300X, ROCm) | gpu-morgeva | 12 h | gpu:1 |

Always use `--account=gpu-research`. n-210 jobs need the `cs_amd` venv
(`/home/morg/students/dortirosh/envs/cs_amd`); everything else uses
`codecslime`.

## Final deliverables (course)

- `project.pdf`: max 5 pages, top of page 1 lists each group member's name
  and ID number. Compiled in Overleaf.
- `project_code.zip`: Python 3.10; `requirements.txt`; `readme.txt` with the
  exact train + eval commands; audio samples from both train and validation
  splits.
