# Melt run on n-210 (AMD MI-class GPUs) — decisions

This doc records the design choices for the n-210 (ROCm) Melt re-run that
replaces the L40s 24h cycle with a single GPU job that should finish in
roughly an eighth of the wall time.

## Hardware

- Node: `n-210`, partition `gpu-morgeva`.
- GRES: `gpu:amd:8`. AMD GPUs (MI-class). 2.3 TB host RAM, 384 CPU threads.
- Per CLAUDE.md the partition convention "t-100 must always use gpu-morgeva"
  does not block us; `gpu-morgeva` is the right partition for n-210 too.

## Environment

`<VENV_PATH>/cs_amd`, fresh build. (The intended name
was `codecslime_amd`; conda recorded it as `cs_amd` and we kept it to avoid
churn. Slurm scripts and docs reference `cs_amd`.)

- Python 3.10 (match the existing `codecslime` env).
- `torch==2.9.1+rocm6.4`, `torchaudio==2.9.1+rocm6.4` from
  `https://download.pytorch.org/whl/rocm6.4` (same wheels validated by the
  existing `olmo2_amd` env).
- `pytorch-lightning==2.5.5` (PL 2.4 from `requirements.txt` is pinned to
  torch<2.5; PL 2.5 supports torch 2.5+ and is API-compatible with the
  trainer / strategy code we use).
- Other deps from `requirements.txt` at their pinned versions
  (numpy 1.26.4, soundfile, librosa, einops, hydra-core, omegaconf,
  vector-quantize-pytorch, wandb, tqdm, pytest).
- Eval-only deps (openai-whisper, jiwer, pystoi, pesq, resemblyzer) are
  intentionally not installed in this env; they pull a torch wheel that
  fights the ROCm pin. They live in the original `codecslime` env (CUDA),
  which we keep for evaluation.

## Batch / LR / step-budget scaling

User instruction: assume the paper batch size is 8, scale LR by sqrt(B/8),
and shrink total steps linearly with B. Concretely:

| Quantity | Paper / L40s | n-210 | Notes |
|---|---|---|---|
| Per-GPU batch | 8 (paper) / 16 (L40s 48GB) | 64 | 4x the L40s. 8x the paper. |
| LR scaling vs paper | 1x | sqrt(64/8) = 2 sqrt(2) ≈ 2.83 | square-root rule |
| `gen_optim_params.max_lr` | 1.0e-4 (paper) | 2.83e-4 | re-warm peak |
| `gen_schedule_params.min_lr` | 1.0e-5 | 2.83e-5 | end of decay |
| `disc_*` mirrors gen | same | same | same schedule for both |
| Total steps | 100000 | 12500 | linear in batch |
| `warmup_step` | 1000 | 125 | keep 1% ratio |
| `down_step` | 99000 | 12375 | total - warmup |
| Melt `s_p` | 100000 | 12500 | curriculum hits `p_tgt` at end |
| Checkpoint cadence | every 5000 | every 625 | linear x1/8, same total count |

Why sqrt: noise scale of SGD/Adam scales like `lr / sqrt(B)`. Holding noise
scale constant when growing B from 8 to 64 means lr grows by sqrt(8). Linear
scaling (8x) on top of an already-tuned 1e-4 would push the model past 8e-4
which is well into territory where the discriminator destabilizes.

Why total steps scale linearly: the paper trains over a fixed number of
**examples**. At 8x larger batch we see the same training data in 1/8 the
steps. We are not changing the data budget.

Why `s_p` follows total steps: the Melt curriculum is parameterized by
"fraction of training progress" via `step / s_p`. We want max randomness
reached at the end of training, regardless of batch size; so `s_p` tracks
total steps.

Warmup of 125 steps is short but plausible (1% of horizon, matches paper
ratio). If we see early instability we can bump to 250 (2%).

## Other choices left at paper defaults

- `p_tgt = [0.1, 0.45, 0.25, 0.2]`, `concentration_control = 30.0`,
  `skip_prob = 0.5`: literal paper values, not touched.
- `USE_PAPER_D_ENFORCE = False`: see project CLAUDE.md.
- Mixed precision: `16-mixed`. AMD MI-class supports bf16 natively but
  16-mixed (fp16 autocast) is what BigCodec / Lightning expect here. If we
  see fp16 underflow on the discriminator loss we can switch to
  `bf16-mixed`.
- Gradient accumulation: off (`accumulate_grad_batches=1`); batch 64 is the
  real microbatch.
- DDP strategy: single GPU per job, so `DDPStrategy(find_unused_parameters=
  True)` reduces to no-op data parallel.

## SLURM

- Two separate jobs, one per backbone (vq8k, fsq18k).
- `--partition=gpu-morgeva`, `--gres=gpu:amd:1`, `--time=12:00:00`.
- No auto-resubmit loop; with 12500 steps we expect to finish inside the
  partition's wall-time bound. If a job dies, the existing
  `auto_resume_melt.sh` will pick up from `last.ckpt`.

## Risk / unknowns

- Exact AMD GPU model: SLURM reports only `gpu:amd:N`. ROCm/torch reports
  the real name on first run. Memory headroom may force a downsize if it
  turns out to be a 32GB part; 64-GB+ MI parts will be fine.
- PL 2.5 vs 2.4 API drift: minor; checkpoint loader on resume might warn
  about saved hyperparameters. Resume from existing L40s checkpoints is
  not planned (separate run, separate checkpoint dir).
