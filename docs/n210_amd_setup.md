# n-210 (AMD MI-series ROCm) setup notes

This documents decisions made when adding an AMD/ROCm runtime path for
Melt-stage training on `n-210`. Original L40s/H100 runs are not affected.

## Hardware
- Node: `n-210` (partition `gpu-morgeva`, GRES `gpu:amd:8`, 8 AMD GPUs, ~2TB RAM)
- One AMD GPU per job is plenty for batch 64 Melt training (paper used batch 8 on H100).

## Conda env: `codecslime_amd`
Location: `/home/morg/students/dortirosh/envs/codecslime_amd`

Differences vs `codecslime` (CUDA):
- `torch==2.9.1+rocm6.4`, `torchaudio==2.9.1+rocm6.4`, `torchvision==0.24.1+rocm6.4`
  from `https://download.pytorch.org/whl/rocm6.4`.
- `pytorch-lightning` bumped from 2.4.0 (codecslime pin) to a torch-2.9-compatible
  release. PL 2.4 is built against torch 2.1-2.4 and won't pip-resolve cleanly
  with torch 2.9. The Lightning API surface used here (Trainer, DDPStrategy,
  ModelCheckpoint, LearningRateMonitor, LightningModule, seed_everything) is
  stable across 2.4 -> 2.5 so this is API-safe.

Everything else (`numpy`, `soundfile`, `librosa`, `einops`, `hydra-core`,
`vector-quantize-pytorch`, `wandb`, `omegaconf`, `tqdm`) is pinned identically
to the CUDA env.

## Batch-64 LR / schedule scaling

Reference (paper / current L40s run): batch=8, max_lr=1e-4, min_lr=1e-5,
warmup=1k, down_step=99k, total_steps=100k, melt s_p=100k.

Scaling rules:
- Effective batch 8 -> 64 (x8).
- Total steps: linear x1/8 -> 12,500.
- LR: square-root scaling for batch x8 -> sqrt(8) ~= 2.828 -> max_lr=2.83e-4,
  min_lr=2.83e-5.
- Warmup: linear x1/8 -> 125 steps.
- Down step: linear x1/8 -> 12,375 steps (warmup + down = 12,500 = total).
- Melt s_p (curriculum length): x1/8 -> 12,500 so max randomness is reached
  at the end of training, same as the paper recipe.
- Checkpoint every_n_train_steps: 5000 -> 625 (x1/8) to keep similar number
  of checkpoints across the run.
