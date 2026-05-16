# Backbone Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two BigCodec-style 80 Hz fixed-frame-rate VQ-GAN speech-codec backbones on LibriSpeech-960h, scaled to 300k steps, producing checkpoints that feed CodecSlime's later Melt-and-Cool stage:

1. **VQ-8k backbone** - BigCodec stock (`ResidualVQ`, codebook 8192, dim 8).
2. **FSQ-18k backbone** - same encoder/decoder/discriminators, quantizer swapped to FSQ (codebook size 18225, projected dim 8, levels `[3,3,3,3,3,3,5,5]`).

**Architecture:** Vendor BigCodec into `audio_ml_tau_final/external/BigCodec/` as a clean upstream clone. Add a small `fsq_quantizer.py` next to BigCodec's `vq/residual_vq.py` and gate the choice with a config key. All CodecSlime-specific configs (W&B logger, scaled step count, dataset paths) live in `audio_ml_tau_final/backbones/configs/` and override BigCodec's defaults via Hydra. Training runs under SLURM with auto-resume across the cluster's 5-day H100 limit, logging to W&B.

**Tech Stack:** Python 3.10, PyTorch + PyTorch Lightning, Hydra, BigCodec (cloned, MIT), `vector-quantize-pytorch` (for FSQ), W&B, SLURM (TAU `gpu-morgeva` / `killable`).

**Verified against paper §3.1.** Every backbone hyperparameter below matches what the paper states or what BigCodec defaults to (paper says "training code is adopted from the official implementation of BigCodec"):

| Parameter | Paper §3.1 | This plan |
|---|---|---|
| Train set | LibriSpeech 960h, 16 kHz, 16-bit | same |
| Token rate | 80 Hz | 80 Hz (encoder up_ratios `[2,2,2,5,5]`, product 200, 16000/200=80) |
| Feature dim `d_h` | 1024 | `out_channels=1024`, `vq_dim=1024` |
| VQ codebook size | 8192 | `codebook_size=8192` |
| FSQ codebook size | 18225 | `levels=[3,3,3,3,3,3,5,5]` (3⁶·5² = 18225) |
| Projected dim | 8 | `codebook_dim=8` (VQ) / `fsq_dim=8` (FSQ) |
| Optimizer | AdamW, β=(0.8, 0.9) | same |
| LR | linear 1e-4 → 1e-5, 1000 warmup | same |
| Total steps (paper) | 1.2M on 2× A800 | scaled to **300k** per user request |
| LR `down_step` | not stated; BigCodec default 500k of 1.2M (41.7%) | **125k** of 300k (preserves 41.7% ratio) |

**Compute and batching, per user decision (this session):**

| Quantity | BigCodec default | Paper | This plan |
|---|---|---|---|
| GPUs per job | 1 | 2× A800 | **1× L40S** (preferred, see fallback below) |
| Batch size per device | 8 | unspecified | 8 |
| Effective global batch | 8 | unspecified | **8** (matches BigCodec exactly) |

**Single-GPU first.** L40S has 48 GB VRAM; BigCodec's stock 8-batch run was tuned for a single GPU and the model is ~159M params. The smoke test (Task 7) confirms whether batch=8 fits on one L40S in `16-mixed` precision. **If OOM: fall back to 4× L40S DDP with `batch_size=2 per device, accumulate_grad_batches=1`** to keep global batch = 8. Both options are covered in the configs.

**FSQ levels.** 18225 = 3^6 * 5^2, so `levels = [3, 3, 3, 3, 3, 3, 5, 5]` (8 dims). All values are within FSQ's recommended range (3-8 per dim). Project encoder feature 1024 -> 8 with `nn.Linear`, FSQ, project 8 -> 1024 back.

---

## File Structure

```
audio_ml_tau_final/
├── external/
│   └── BigCodec/                      # cloned upstream, gitignored
│       └── vq/
│           └── fsq_quantizer.py       # NEW - small wrapper around vector-quantize-pytorch FSQ
├── backbones/
│   ├── __init__.py
│   ├── configs/
│   │   ├── codecslime_vq8k.yaml       # Hydra entry point for VQ run
│   │   ├── codecslime_fsq18k.yaml     # Hydra entry point for FSQ run
│   │   ├── dataset/
│   │   │   └── librispeech.yaml
│   │   ├── model/
│   │   │   ├── vq8k.yaml              # codec_decoder uses ResidualVQ
│   │   │   └── fsq18k.yaml            # codec_decoder uses FSQ
│   │   └── train/
│   │       └── codecslime_300k.yaml   # max_steps=300000, down_step=125000, W&B logger
│   ├── scripts/
│   │   ├── prepare_librispeech.py     # downloads 960h + dev-clean, runs BigCodec preprocess
│   │   └── smoke_train.sh             # 100-step single-GPU sanity run
│   ├── slurm/
│   │   ├── train_vq8k.slurm
│   │   ├── train_fsq18k.slurm
│   │   └── auto_resume.sh             # resubmit-on-exit wrapper, finds last.ckpt
│   └── tests/
│       ├── test_fsq_codebook_size.py  # unique-code count == 18225
│       ├── test_fsq_forward_shape.py
│       └── test_decoder_swap.py       # CodecDecoder builds with quantizer=fsq
└── docs/
    └── plans/
        └── 2026-05-09-backbone-training.md  # this file
```

LibriSpeech goes **inside the project** at `audio_ml_tau_final/datasets/LibriSpeech/` (gitignored; raw audio, ~85 GB uncompressed). Project must be 100% self-contained per project requirements - all paths in configs are relative to the repo root or absolute under it.

W&B project: `codecslime-backbones` under entity `dortiroshtau-tel-aviv-university`. Run names: `vq8k-300k` and `fsq18k-300k`.

---

## Task 1: Vendor BigCodec into the repo

**Files:**
- Create: `audio_ml_tau_final/external/.gitkeep`
- Modify: `audio_ml_tau_final/.gitignore`

- [ ] **Step 1: Add `external/BigCodec/` to .gitignore**

Append to `audio_ml_tau_final/.gitignore`:

```
# vendored upstream code, kept out of git
external/BigCodec/
```

- [ ] **Step 2: Create `external/` placeholder so the dir is tracked**

```bash
mkdir -p /home/morg/students/dortirosh/audio_ml_tau_final/external
touch /home/morg/students/dortirosh/audio_ml_tau_final/external/.gitkeep
```

- [ ] **Step 3: Clone BigCodec at a pinned commit**

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final/external
git clone https://github.com/Aria-K-Alethia/BigCodec.git
cd BigCodec
git rev-parse HEAD > /home/morg/students/dortirosh/audio_ml_tau_final/external/BIGCODEC_COMMIT.txt
```

- [ ] **Step 4: Verify the expected files exist**

```bash
ls /home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec/{train.py,preprocess.py,lightning_module.py,vq/codec_decoder.py,config/default.yaml}
```

Expected: all five paths print without error.

- [ ] **Step 5: Commit**

```bash
git add .gitignore external/.gitkeep external/BIGCODEC_COMMIT.txt
git commit -m "vendor BigCodec under external/ (gitignored)"
```

---

## Task 2: Python 3.10 environment + dependencies

**Files:**
- Create: `audio_ml_tau_final/requirements.txt`
- Create: `audio_ml_tau_final/backbones/scripts/setup_env.sh`

- [ ] **Step 1: Pick env path and create the venv**

```bash
python3.10 -m venv /home/morg/students/dortirosh/envs/codecslime
source /home/morg/students/dortirosh/envs/codecslime/bin/activate
python --version  # expect Python 3.10.x
pip install --upgrade pip
```

- [ ] **Step 2: Write `requirements.txt`**

Replace the existing single-line `requirements.txt` with:

```
numpy==1.26.4
soundfile==0.12.1
librosa==0.10.2
einops==0.8.0
torch==2.4.0
torchaudio==2.4.0
pytorch-lightning==2.4.0
hydra-core==1.3.2
vector-quantize-pytorch==1.17.3
wandb==0.18.3
omegaconf==2.3.0
tqdm==4.66.5
```

Pinned versions: BigCodec is light on version constraints; pinning here avoids future breakage. `vector-quantize-pytorch` provides the FSQ implementation we'll use.

- [ ] **Step 3: Install**

```bash
pip install -r /home/morg/students/dortirosh/audio_ml_tau_final/requirements.txt
```

- [ ] **Step 4: Verify imports**

Run:

```bash
python -c "import torch, torchaudio, pytorch_lightning, hydra, vector_quantize_pytorch, wandb, soundfile, librosa, einops; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected: prints versions, `cuda True` (if run on a GPU node) or `cuda False` (if on a login node - that's also OK).

- [ ] **Step 5: W&B login**

```bash
wandb login   # paste API key once; persists in ~/.netrc
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "pin training-stack requirements (Python 3.10, torch 2.4, lightning 2.4)"
```

---

## Task 3: Download LibriSpeech and build BigCodec filelists

**Files:**
- Create: `audio_ml_tau_final/backbones/scripts/prepare_librispeech.py`

- [ ] **Step 1: Reserve a dataset directory inside the project**

```bash
mkdir -p /home/morg/students/dortirosh/audio_ml_tau_final/datasets
df -h /home/morg/students/dortirosh/audio_ml_tau_final/datasets/   # confirm > 100 GB free
echo "datasets/" >> /home/morg/students/dortirosh/audio_ml_tau_final/.gitignore
```

- [ ] **Step 2: Write the download script**

Create `audio_ml_tau_final/backbones/scripts/prepare_librispeech.py`:

```python
"""Download LibriSpeech subsets to a target dir and extract."""
import argparse
import subprocess
from pathlib import Path

URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="parent dir; LibriSpeech/ will appear inside")
    ap.add_argument("--subsets", nargs="+", default=list(URLS.keys()))
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.subsets:
        tarball = args.root / f"{name}.tar.gz"
        marker = args.root / "LibriSpeech" / name
        if marker.exists():
            print(f"[skip] {name} already extracted")
            continue
        if not tarball.exists():
            print(f"[get] {name}")
            subprocess.check_call(["wget", "-c", URLS[name], "-O", str(tarball)])
        print(f"[extract] {name}")
        subprocess.check_call(["tar", "-xzf", str(tarball), "-C", str(args.root)])

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run download (this takes ~1-2 hours on a fast link)**

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final
python backbones/scripts/prepare_librispeech.py \
    --root /home/morg/students/dortirosh/audio_ml_tau_final/datasets
```

Expected ending:
```
[extract] test-clean
```

- [ ] **Step 4: Sanity-check the extracted layout**

```bash
ls /home/morg/students/dortirosh/audio_ml_tau_final/datasets/LibriSpeech/
# expect: train-clean-100  train-clean-360  train-other-500  dev-clean  test-clean
find /home/morg/students/dortirosh/audio_ml_tau_final/datasets/LibriSpeech/train-clean-100 -name "*.flac" | head -3
# expect: three FLAC paths
```

- [ ] **Step 5: Build the BigCodec filelists**

BigCodec's `preprocess.py` writes a flat list of `.flac` paths. Run it through Hydra, pointing at our in-project root:

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec
mkdir -p filelists
python preprocess.py \
    hydra.output_subdir=null hydra.job.chdir=False \
    preprocess.datasets.LibriSpeech.root=/home/morg/students/dortirosh/audio_ml_tau_final/datasets/LibriSpeech \
    preprocess.view.train_filelist=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_train.txt \
    preprocess.view.test_filelist=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_test.txt
```

Then:
```bash
mkdir -p /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data
wc -l /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_train.txt
# expect ~281k lines (100h: ~28k + 360h: ~104k + 500h: ~149k)
wc -l /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_test.txt
# expect ~2700 lines (dev-clean)
```

- [ ] **Step 6: Commit**

```bash
git add backbones/scripts/prepare_librispeech.py
# do NOT commit backbones/data/*.txt - they reference machine-local paths;
# instead ensure they're gitignored
echo "backbones/data/" >> .gitignore
git add .gitignore
git commit -m "add LibriSpeech download script and gitignore filelists"
```

---

## Task 4: CodecSlime-specific Hydra configs

**Files:**
- Create: `audio_ml_tau_final/backbones/configs/codecslime_vq8k.yaml`
- Create: `audio_ml_tau_final/backbones/configs/codecslime_fsq18k.yaml`
- Create: `audio_ml_tau_final/backbones/configs/dataset/librispeech.yaml`
- Create: `audio_ml_tau_final/backbones/configs/model/vq8k.yaml`
- Create: `audio_ml_tau_final/backbones/configs/model/fsq18k.yaml`
- Create: `audio_ml_tau_final/backbones/configs/train/codecslime_300k.yaml`

- [ ] **Step 1: Write `dataset/librispeech.yaml`**

```yaml
dataset:
  _target_: data_module.FSDataset

train:
  filelist: /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_train.txt
  batch_size: 8
  shuffle: true

val:
  filelist: /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_test.txt
  batch_size: 8
  shuffle: false

test:
  filelist: /home/morg/students/dortirosh/audio_ml_tau_final/backbones/data/librispeech_test.txt
  batch_size: 1
  shuffle: false

padding_idx: 0
frame_length: 200
min_audio_length: 16000
```

- [ ] **Step 2: Write `model/vq8k.yaml`** (same as BigCodec default; explicit for clarity)

```yaml
codec_encoder:
  out_channels: 1024
  ngf: 48
  use_rnn: True
  rnn_bidirectional: False
  rnn_num_layers: 2
  up_ratios: [2, 2, 2, 5, 5]
  dilations: [1, 3, 9]

codec_decoder:
  quantizer_type: vq
  in_channels: 1024
  upsample_initial_channel: 1536
  ngf: 48
  use_rnn: True
  rnn_bidirectional: False
  rnn_num_layers: 2
  up_ratios: [5, 5, 2, 2, 2]
  dilations: [1, 3, 9]
  vq_num_quantizers: 1
  vq_dim: 1024
  vq_commit_weight: 0.25
  vq_weight_init: False
  vq_full_commit_loss: False
  codebook_size: 8192
  codebook_dim: 8

mpd:
  periods: [2, 3, 5, 7, 11]
  max_downsample_channels: 512
  channels: 16
  channel_increasing_factor: 4

mstft:
  stft_params:
    fft_sizes: [128, 256, 512, 1024, 2048]
    hop_sizes: [32, 64, 128, 256, 512]
    win_lengths: [128, 256, 512, 1024, 2048]
    window: hann_window
  in_channels: 1
  out_channels: 1
  kernel_sizes: [5, 3]
  channels: 32
  max_downsample_channels: 512
  downsample_scales: [2, 2, 2]
  use_weight_norm: True
```

The new key here is `quantizer_type: vq`. Task 5 reads this in `codec_decoder.py`.

- [ ] **Step 3: Write `model/fsq18k.yaml`**

Same as `vq8k.yaml` but the `codec_decoder` block becomes:

```yaml
codec_decoder:
  quantizer_type: fsq
  in_channels: 1024
  upsample_initial_channel: 1536
  ngf: 48
  use_rnn: True
  rnn_bidirectional: False
  rnn_num_layers: 2
  up_ratios: [5, 5, 2, 2, 2]
  dilations: [1, 3, 9]
  vq_dim: 1024
  fsq_levels: [3, 3, 3, 3, 3, 3, 5, 5]   # prod = 18225
  fsq_dim: 8
```

(The remaining `mpd`/`mstft`/`codec_encoder` blocks are identical to `vq8k.yaml`. Copy them verbatim.)

- [ ] **Step 4: Write `train/codecslime_300k.yaml`**

Single-GPU by default; if smoke test (Task 7) shows OOM, switch `devices: 1` to `devices: 4` and `dataset.train.batch_size: 8` to `dataset.train.batch_size: 2` to keep global batch = 8.

```yaml
trainer:
  accelerator: 'gpu'
  devices: 1
  min_steps: 300000
  max_steps: 300000
  precision: 16-mixed
  limit_val_batches: 0
  num_sanity_val_steps: 0
  accumulate_grad_batches: 1

lambdas:
  lambda_disc: 1.0
  lambda_feat_match_loss: 1.0
  lambda_mel_loss: 15.0
  lambda_adv: 1.0
  lambda_stft_loss: 1.0

use_mel_loss: true
use_feat_match_loss: true

stft_loss_params:
  fft_sizes: [128, 256, 512, 1024, 2048]
  hop_sizes: [32, 64, 128, 256, 512]
  win_lengths: [128, 256, 512, 1024, 2048]
  window: hann_window

gen_optim_params:
  lr: 1.0
  betas: [0.8, 0.9]
disc_optim_params:
  lr: 1.0
  betas: [0.8, 0.9]
gen_grad_clip: 1.0
disc_grad_clip: 1.0

gen_schedule_params:
  warmup_step: 1000
  down_step: 125000
  min_lr: 1.0e-5
  max_lr: 1.0e-4
disc_schedule_params:
  warmup_step: 1000
  down_step: 125000
  min_lr: 1.0e-5
  max_lr: 1.0e-4

logger:
  _target_: pytorch_lightning.loggers.WandbLogger
  project: codecslime-backbones
  entity: dortiroshtau-tel-aviv-university
  name: ???    # set per-run via CLI override
  resume: allow
  id: ???      # set per-run via CLI override

checkpoint:
  every_n_train_steps: 5000
  save_top_k: 3
  save_last: true
```

`logger.name` and `logger.id` get filled by the launcher script (Task 7) to support resumable W&B runs.

- [ ] **Step 5: Write `codecslime_vq8k.yaml`** (entry-point that composes the above)

```yaml
defaults:
  - preprocess: default
  - dataset: librispeech
  - model: vq8k
  - train: codecslime_300k

log_dir: /home/morg/students/dortirosh/audio_ml_tau_final/backbones/checkpoints/vq8k
debug: false
ckpt: null
input_dir: null
output_dir: null

hydra:
  output_subdir: hydra
  job:
    chdir: True
```

- [ ] **Step 6: Write `codecslime_fsq18k.yaml`** (same but `model: fsq18k` and `log_dir: .../fsq18k`)

```yaml
defaults:
  - preprocess: default
  - dataset: librispeech
  - model: fsq18k
  - train: codecslime_300k

log_dir: /home/morg/students/dortirosh/audio_ml_tau_final/backbones/checkpoints/fsq18k
debug: false
ckpt: null
input_dir: null
output_dir: null

hydra:
  output_subdir: hydra
  job:
    chdir: True
```

- [ ] **Step 7: Sym-link the new config dirs into BigCodec so Hydra finds them**

BigCodec's `train.py` calls `@hydra.main(version_base=None, config_path="config", config_name="default")` from inside the BigCodec dir. We override `config_path` via `--config-dir` at CLI, so no symlinking needed. Verify Hydra discovery now:

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec
python -c "
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir='/home/morg/students/dortirosh/audio_ml_tau_final/backbones/configs', version_base=None):
    cfg = compose(config_name='codecslime_vq8k')
print(cfg.train.trainer.max_steps, cfg.model.codec_decoder.codebook_size)
"
```

Expected: `300000 8192`

- [ ] **Step 8: Commit**

```bash
git add backbones/configs/
git commit -m "add CodecSlime backbone configs (VQ8k, FSQ18k, 300k schedule)"
```

---

## Task 5: FSQ quantizer module

**Files:**
- Create: `audio_ml_tau_final/external/BigCodec/vq/fsq_quantizer.py`
- Create: `audio_ml_tau_final/backbones/tests/test_fsq_codebook_size.py`
- Create: `audio_ml_tau_final/backbones/tests/test_fsq_forward_shape.py`

The FSQ wrapper has the same call surface as `ResidualVQ`'s forward in `codec_decoder.py`, returning `(quantized_features, indices, commit_loss)` where `commit_loss` is always a zero scalar (FSQ has no codebook to learn).

- [ ] **Step 1: Write the failing test for unique-code count**

`audio_ml_tau_final/backbones/tests/test_fsq_codebook_size.py`:

```python
"""FSQ with levels [3,3,3,3,3,3,5,5] should yield exactly 18225 codes."""
import sys
import torch
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

from vq.fsq_quantizer import FSQQuantizer

def test_codebook_size_18225():
    q = FSQQuantizer(input_dim=1024, fsq_dim=8, levels=[3, 3, 3, 3, 3, 3, 5, 5])
    assert q.codebook_size == 18225
```

- [ ] **Step 2: Run the test, expect ImportError**

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final
pytest backbones/tests/test_fsq_codebook_size.py -v
```

Expected: `ModuleNotFoundError: No module named 'vq.fsq_quantizer'`.

- [ ] **Step 3: Write `fsq_quantizer.py`**

```python
"""FSQ wrapper that mimics ResidualVQ's call surface inside CodecDecoder."""
from math import prod
import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ


class FSQQuantizer(nn.Module):
    def __init__(self, input_dim: int, fsq_dim: int, levels):
        super().__init__()
        self.levels = list(levels)
        self.codebook_size = prod(self.levels)
        self.proj_in = nn.Linear(input_dim, fsq_dim)
        self.proj_out = nn.Linear(fsq_dim, input_dim)
        self.fsq = FSQ(levels=self.levels)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C) - matches ResidualVQ input convention used by CodecDecoder
        z = self.proj_in(x)
        z_q, indices = self.fsq(z)
        out = self.proj_out(z_q)
        commit_loss = x.new_zeros(())  # FSQ has no learnable codebook
        return out, indices, commit_loss
```

- [ ] **Step 4: Run codebook-size test, expect PASS**

```bash
pytest backbones/tests/test_fsq_codebook_size.py -v
```

- [ ] **Step 5: Add forward-shape test**

`audio_ml_tau_final/backbones/tests/test_fsq_forward_shape.py`:

```python
import sys
import torch
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

from vq.fsq_quantizer import FSQQuantizer

def test_forward_shape_and_indices_range():
    torch.manual_seed(0)
    q = FSQQuantizer(input_dim=1024, fsq_dim=8, levels=[3, 3, 3, 3, 3, 3, 5, 5])
    x = torch.randn(2, 50, 1024)
    out, indices, commit_loss = q(x)
    assert out.shape == (2, 50, 1024)
    assert indices.shape == (2, 50)
    assert indices.min().item() >= 0
    assert indices.max().item() < 18225
    assert commit_loss.item() == 0.0
```

- [ ] **Step 6: Run, expect PASS**

```bash
pytest backbones/tests/test_fsq_forward_shape.py -v
```

- [ ] **Step 7: Commit**

```bash
git add external/BigCodec/vq/fsq_quantizer.py backbones/tests/test_fsq_*.py
# external/BigCodec is gitignored - force-add the one new file we own
git add -f external/BigCodec/vq/fsq_quantizer.py
git commit -m "add FSQ quantizer with 18225-code wrapper and tests"
```

---

## Task 6: Wire `quantizer_type` into CodecDecoder

**Files:**
- Modify: `audio_ml_tau_final/external/BigCodec/vq/codec_decoder.py`
- Create: `audio_ml_tau_final/backbones/tests/test_decoder_swap.py`

CodecDecoder hardcodes `ResidualVQ`. We add a single branch keyed off `quantizer_type` (default "vq" preserves existing behaviour).

- [ ] **Step 1: Read the current decoder to find exact line numbers**

```bash
grep -n "ResidualVQ\|def __init__\|def forward\|quantizer" \
    /home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec/vq/codec_decoder.py
```

Note the line numbers for `ResidualVQ(...)` instantiation and `def __init__`.

- [ ] **Step 2: Write the failing swap test**

`audio_ml_tau_final/backbones/tests/test_decoder_swap.py`:

```python
"""CodecDecoder must accept quantizer_type='fsq' and produce the right output shape."""
import sys
import torch
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

from vq.codec_decoder import CodecDecoder


def test_vq_decoder_default():
    dec = CodecDecoder()  # quantizer_type defaults to "vq"
    assert hasattr(dec, "quantizer")


def test_fsq_decoder_swap():
    dec = CodecDecoder(
        quantizer_type="fsq",
        fsq_levels=[3, 3, 3, 3, 3, 3, 5, 5],
        fsq_dim=8,
    )
    x = torch.randn(1, 1024, 80)  # (B, C, T) - the encoder output convention
    out, indices, commit = dec(x, vq=True)
    # decoder upsamples 200x: 80 frames -> 16000 samples
    assert out.shape[-1] == 80 * 200
    assert indices.shape[-1] == 80
    assert commit.item() == 0.0
```

- [ ] **Step 3: Run, expect FAIL** (`unexpected keyword argument 'quantizer_type'`)

```bash
pytest backbones/tests/test_decoder_swap.py -v
```

- [ ] **Step 4: Patch CodecDecoder to branch on `quantizer_type`**

In `external/BigCodec/vq/codec_decoder.py`, modify `__init__` to:

1. Accept new kwargs: `quantizer_type: str = "vq"`, `fsq_levels: list | None = None`, `fsq_dim: int | None = None`.
2. Replace the hardcoded `self.quantizer = ResidualVQ(...)` block with:

```python
if quantizer_type == "vq":
    from .residual_vq import ResidualVQ
    self.quantizer = ResidualVQ(
        num_quantizers=vq_num_quantizers,
        dim=vq_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        threshold_ema_dead_code=2,
        commitment=vq_commit_weight,
        weight_init=vq_weight_init,
        full_commit_loss=vq_full_commit_loss,
    )
elif quantizer_type == "fsq":
    from .fsq_quantizer import FSQQuantizer
    assert fsq_levels is not None and fsq_dim is not None, \
        "fsq_levels and fsq_dim required when quantizer_type='fsq'"
    self.quantizer = FSQQuantizer(input_dim=vq_dim, fsq_dim=fsq_dim, levels=fsq_levels)
else:
    raise ValueError(f"unknown quantizer_type: {quantizer_type}")
```

`forward(x, vq=True)` already returns `(out, q, commit)` from `self.quantizer(...)`, so no changes there - both `ResidualVQ` and `FSQQuantizer` honor the same return signature.

Important: `ResidualVQ` and `FSQQuantizer` may take input in different layouts (channels-first vs channels-last). Check the existing decoder's call site for `self.quantizer(...)` - if it transposes from `(B, C, T)` to `(B, T, C)` before calling and back after, FSQ already handles that. If not, add `.transpose(1, 2)` before/after the FSQ call inside the `elif` branch's wrapper.

- [ ] **Step 5: Run swap test, expect PASS**

```bash
pytest backbones/tests/test_decoder_swap.py -v
```

- [ ] **Step 6: Commit**

```bash
git add -f external/BigCodec/vq/codec_decoder.py
git add backbones/tests/test_decoder_swap.py
git commit -m "wire quantizer_type into CodecDecoder for FSQ swap"
```

---

## Task 7: Smoke test (single GPU, 100 steps)

**Files:**
- Create: `audio_ml_tau_final/backbones/scripts/smoke_train.sh`

This catches config errors, dataloader bugs, OOMs, and W&B-init issues before the long SLURM job goes out.

- [ ] **Step 1: Write the smoke script**

```bash
#!/bin/bash
# 100-step single-GPU sanity run. Validates the full plumbing.
set -euo pipefail

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec

source /home/morg/students/dortirosh/envs/codecslime/bin/activate
export PYTHONPATH=$BIGCODEC:$PYTHONPATH

cd $BIGCODEC
python train.py \
    --config-dir=$REPO/backbones/configs \
    --config-name=codecslime_vq8k \
    train.trainer.devices=1 \
    train.trainer.max_steps=100 \
    train.trainer.min_steps=100 \
    dataset.train.batch_size=2 \
    train.checkpoint.every_n_train_steps=50 \
    train.logger.name=smoke-vq8k \
    train.logger.id=smoke-vq8k \
    log_dir=$REPO/backbones/checkpoints/smoke
```

Save as `audio_ml_tau_final/backbones/scripts/smoke_train.sh` and `chmod +x` it.

- [ ] **Step 2: Run on a free L40S**

Reserve one L40S interactively. This both validates the pipeline AND confirms whether batch=8 fits on a single L40S (the 1-GPU-vs-4-GPU decision):

```bash
srun --partition=killable --account=gpu-research --gres=gpu:l40s:1 \
     --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash
$REPO/backbones/scripts/smoke_train.sh
```

If this OOMs: re-edit `backbones/configs/train/codecslime_300k.yaml` to `devices: 4` and `dataset.train.batch_size: 2`, update the SLURM scripts (Task 8) to `--gres=gpu:l40s:4`, and re-run the smoke.

Expected ending (approximately):
```
Epoch 0: 100/100 [..] mel_loss=... gen_loss=... disc_loss=...
Training ends, best score: ..., ckpt path: .../smoke/last.ckpt
```

- [ ] **Step 3: Verify W&B saw it**

Open `https://wandb.ai/dortiroshtau-tel-aviv-university/codecslime-backbones`, confirm the `smoke-vq8k` run shows ~100 logged steps with mel_loss, disc_loss, gen_loss, and learning rate curves.

- [ ] **Step 4: Repeat for FSQ**

```bash
$REPO/backbones/scripts/smoke_train.sh   # but edit to use codecslime_fsq18k.yaml
```

Or pass via env var; up to you. The point is to catch FSQ-specific surprises (e.g., layout mismatch) before the long run.

- [ ] **Step 5: Commit**

```bash
git add backbones/scripts/smoke_train.sh
git commit -m "add 100-step smoke test runner"
```

---

## Task 8: SLURM auto-resume launcher

**Files:**
- Create: `audio_ml_tau_final/backbones/slurm/auto_resume.sh`
- Create: `audio_ml_tau_final/backbones/slurm/train_vq8k.slurm`
- Create: `audio_ml_tau_final/backbones/slurm/train_fsq18k.slurm`

Pattern: each `.slurm` script runs the launcher with a fixed run name. The launcher (a) finds the last checkpoint in `log_dir`, (b) uses the same W&B run id every time so resumption appends rather than forking, (c) self-resubmits on graceful exit if `max_steps` not reached.

- [ ] **Step 1: Write `auto_resume.sh`**

```bash
#!/bin/bash
# Resume-aware launcher. Args:
#   $1 = config name (codecslime_vq8k | codecslime_fsq18k)
#   $2 = run name / W&B id  (e.g. vq8k-300k)
set -euo pipefail

CFG=$1
RUN=$2

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
LOG_DIR=$REPO/backbones/checkpoints/$RUN

source /home/morg/students/dortirosh/envs/codecslime/bin/activate
export PYTHONPATH=$BIGCODEC:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$LOG_DIR"

# Find last checkpoint, if any
LAST_CKPT=""
if [ -f "$LOG_DIR/last.ckpt" ]; then
    LAST_CKPT="$LOG_DIR/last.ckpt"
    echo "[auto_resume] resuming from $LAST_CKPT"
fi

cd "$BIGCODEC"
RESUME_ARG=""
[ -n "$LAST_CKPT" ] && RESUME_ARG="ckpt=$LAST_CKPT"

python train.py \
    --config-dir="$REPO/backbones/configs" \
    --config-name="$CFG" \
    train.logger.name="$RUN" \
    train.logger.id="$RUN" \
    log_dir="$LOG_DIR" \
    $RESUME_ARG
```

`chmod +x backbones/slurm/auto_resume.sh`.

Note: BigCodec's `train.py` reads `cfg.ckpt`; we'll need to verify that path actually triggers Lightning's `Trainer.fit(..., ckpt_path=cfg.ckpt)`. Check `external/BigCodec/train.py` line ~`trainer.fit(...)` and add `ckpt_path=cfg.ckpt if cfg.ckpt else None` if it isn't already there. If it is, this just works.

- [ ] **Step 2: Write `train_vq8k.slurm`** (1× L40S, killable 1-day partition, auto-resubmit)

```bash
#!/bin/bash
#SBATCH --job-name=cs-vq8k
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120

REPO=/home/morg/students/dortirosh/audio_ml_tau_final

# Resubmit ourselves before SLURM kills us, so checkpointing has time to finish
trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume.sh codecslime_vq8k vq8k-300k
```

If smoke test forced the 4-GPU fallback, change `--gres=gpu:l40s:1` to `--gres=gpu:l40s:4`, `--cpus-per-task=8` to `32`, and `--mem=64G` to `256G`.

- [ ] **Step 3: Write `train_fsq18k.slurm`** (identical except names)

```bash
#!/bin/bash
#SBATCH --job-name=cs-fsq18k
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120

REPO=/home/morg/students/dortirosh/audio_ml_tau_final

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume.sh codecslime_fsq18k fsq18k-300k
```

(Same 4× L40S adjustment applies if needed.)

- [ ] **Step 4: Verify the launcher in dry-run mode**

```bash
bash -n /home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/auto_resume.sh
bash -n /home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/train_vq8k.slurm
bash -n /home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/train_fsq18k.slurm
```

Expected: no output (syntax OK).

- [ ] **Step 5: Commit**

```bash
mkdir -p backbones/slurm/logs
echo "backbones/slurm/logs/" >> .gitignore
git add .gitignore backbones/slurm/auto_resume.sh \
        backbones/slurm/train_vq8k.slurm backbones/slurm/train_fsq18k.slurm
git commit -m "add SLURM auto-resume launchers for VQ8k and FSQ18k"
```

---

## Task 9: Submit and monitor

- [ ] **Step 1: Confirm L40S availability**

```bash
sinfo -p killable -o "%n %G %t %a" | grep l40s
squeue -u $USER
```

L40S nodes are n-801..n-805 with 8 GPUs each, so two single-GPU jobs typically run in parallel. The `killable` partition is preemptible with a 24h limit, so the auto-resume trap is what carries jobs across multi-day training.

- [ ] **Step 2: Submit VQ8k**

```bash
cd /home/morg/students/dortirosh/audio_ml_tau_final
sbatch backbones/slurm/train_vq8k.slurm
```

Note the job id, e.g. `123456`. Watch:

```bash
squeue -u $USER
tail -f backbones/slurm/logs/cs-vq8k_123456.out
```

Expected within ~5 min: `Epoch 0: 0/... loss=...` lines start streaming.

- [ ] **Step 3: Submit FSQ18k**

```bash
sbatch backbones/slurm/train_fsq18k.slurm
```

- [ ] **Step 4: Monitor W&B for 24 hours**

Check that:
- mel_loss declines roughly monotonically
- disc_loss / gen_loss oscillate but stay bounded (typical GAN behaviour, not exploding)
- LR schedule matches: warmup ramp 0 -> 1e-4 over 1000 steps, then linear decay over 125k steps to 1e-5
- step rate roughly 1-2 it/s on 1× L40S (i.e. 300k steps in 2-4 days, with multiple SLURM resubmissions)

If anything is off, kill, fix, restart from last checkpoint.

- [ ] **Step 5: Mark backbone training complete when both runs hit step 300000**

Final checkpoints will be at:
- `backbones/checkpoints/vq8k-300k/last.ckpt`
- `backbones/checkpoints/fsq18k-300k/last.ckpt`

These are the two FFR backbones the CodecSlime Melt-and-Cool stage will fine-tune.

---

## Self-Review

**Spec coverage check:**
- VQ-8k backbone: tasks 1-4, 7-9 cover environment, data, config, smoke, train. ✅
- FSQ-18k backbone: tasks 5-6 add the quantizer; tasks 4, 7-9 share infra. ✅
- 80 Hz FFR: encoder/decoder up_ratios `[2,2,2,5,5]` give 200x downsampling at 16 kHz = 80 Hz. ✅
- LibriSpeech 960h: task 3 downloads all three train subsets. ✅
- BigCodec as basis: tasks 1, 5, 6 vendor and minimally extend. ✅
- W&B logging: built into `train/codecslime_300k.yaml`, set per-run by launcher. ✅
- SLURM auto-resume: tasks 7-8 cover. ✅

**Placeholder scan:** No "TBD" / "TODO" / "fill in" remaining. Each step has executable code or commands.

**Type / API consistency:**
- `FSQQuantizer.forward` returns `(out, indices, commit_loss)` - same arity as `ResidualVQ`'s. ✅
- `quantizer_type` config key is consistent across `model/vq8k.yaml`, `model/fsq18k.yaml`, and the `CodecDecoder.__init__` patch. ✅
- W&B `entity`, `project`, run names consistent across train config and SLURM launchers. ✅

**One open verification (do during Task 8 Step 1):** BigCodec's `train.py` calls `trainer.fit(...)` - we need to confirm whether it forwards `cfg.ckpt` as `ckpt_path`. If not, add a one-line patch there. This is called out in Task 8 Step 1 itself.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-05-09-backbone-training.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
