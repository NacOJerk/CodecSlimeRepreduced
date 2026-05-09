# Backbone Training Appendix

This document records the exact hyperparameters used to train the two FFR
backbones (VQ-8k, FSQ-18k) for our CodecSlime re-implementation, and notes
every place we deviated from the paper's recipe along with the reason.

The paper of record is Wang et al. 2025, *CodecSlime: Temporal Redundancy
Compression of Neural Speech Codec via Dynamic Frame Rate*, arXiv 2506.21074.
Reference: `papers/codecslime_2506.21074.pdf`. The paper says the training
code is "adopted from the official implementation of BigCodec" (Xin et al.,
arXiv 2409.05377). We vendor BigCodec at `external/BigCodec/`, commit
`09845ab1f5bc7a3d1589c16de820ef15bc2afefe`.

## Dataset

**Train.** Full LibriSpeech 960h training set, identical to the paper:
`train-clean-100` (~28k utterances, 100 h, ~6.3 GB), `train-clean-360`
(~104k, 360 h, ~23 GB), `train-other-500` (~149k, 500 h, ~30 GB). Audio is
16 kHz 16-bit FLAC. Total ~281k utterances.

**Test / dev.** `dev-clean` (~2.7k utterances) is used as the validation
filelist by BigCodec's data module, but our training config sets
`limit_val_batches: 0` and `num_sanity_val_steps: 0` so validation is never
run during training. `test-clean` is downloaded for downstream evaluation.

**Storage.** Raw FLACs live at
`audio_ml_tau_final/datasets/LibriSpeech/<subset>/<speaker>/<chapter>/<id>.flac`.
The directory is gitignored. The training file lists are produced by
BigCodec's `preprocess.py` and stored at `backbones/data/librispeech_*.txt`
(also gitignored, machine-local paths).

**Audio loading.** Each `__getitem__` call randomly crops a 1-second
(`min_audio_length=16000` samples) segment from a randomly drawn FLAC. At
80 Hz token rate this is 80 frames per training example. Frames per token
= 200 = product of encoder up_ratios `[2, 2, 2, 5, 5]` at 16 kHz.

## Architecture (matches paper and BigCodec defaults exactly)

Backbone is a VQ-GAN with a CNN+LSTM encoder, a quantizer, a CNN+LSTM
decoder, and two discriminators (HiFiGAN MPD plus a multi-STFT spec
discriminator). About 159M trainable parameters end-to-end.

| Block | Setting | Source |
|---|---|---|
| Encoder out / decoder in channels (`d_h`) | 1024 | paper §3.1 |
| Encoder up_ratios | `[2, 2, 2, 5, 5]` | BigCodec default |
| Decoder up_ratios | `[5, 5, 2, 2, 2]` | BigCodec default |
| Encoder/decoder LSTM | 2 layers, unidirectional | BigCodec default |
| Decoder upsample initial channels | 1536 | BigCodec default |
| `ngf` | 48 | BigCodec default |
| Token frame rate | 80 Hz at 16 kHz audio | paper §3.1 |
| MPD periods | `[2, 3, 5, 7, 11]` | BigCodec default |
| Multi-STFT FFT/hop/win | `[128, 256, 512, 1024, 2048]` triplets | BigCodec default |

### Quantizer (the only place VQ-8k and FSQ-18k differ)

| Variant | Quantizer | Codebook | Projected dim | Source |
|---|---|---|---|---|
| VQ-8k | `ResidualVQ` (single quantizer, EMA dead-code threshold 2, commit weight 0.25) | 8192 | 8 | paper §3.1 + BigCodec default |
| FSQ-18k | `vector_quantize_pytorch.FSQ` with `levels=[3,3,3,3,3,3,5,5]` (3⁶·5² = 18225) | 18225 | 8 | paper §3.1; level decomposition is ours, see "Note 5" below |

### Loss

Manual generator/discriminator alternation (PyTorch Lightning
`automatic_optimization=False`). Total generator loss is

```
L_gen = 15.0 * mel_loss + 1.0 * adv_loss + 1.0 * fm_loss + vq_commit_loss
```

(plus the spec-discriminator's feature-matching contribution). Discriminator
loss is the sum over MPD and Spec branches of LSGAN real+fake loss with
weight 1.0. STFT-loss weight is configured but `use_mel_loss: true` /
`use_feat_match_loss: true` are both on. All loss weights match BigCodec
defaults; the paper does not list them.

## Optimization

| Hyperparameter | Value | Source |
|---|---|---|
| Optimizer | AdamW, betas=(0.8, 0.9) | paper §3.1 |
| Gradient clip (gen and disc) | 1.0 (norm) | BigCodec default |
| Mixed precision | `16-mixed` | BigCodec default |
| Warmup steps | 1000 | paper §3.1 |
| LR schedule shape | linear decay then flat at min_lr | BigCodec default |
| Batch size per GPU (microbatch) | **32** | derived; see Note 1 |
| GPUs per job | **1× H100 80GB** | derived; see Note 2 |
| Effective global batch | **32** | (1 GPU × 32) |
| `accumulate_grad_batches` | 1 | BigCodec default |
| Max LR | **2.0e-4** | derived; see Note 3 |
| Min LR | **2.0e-5** | derived; see Note 3 |
| LR `down_step` | 125,000 | derived; see Note 4 |
| `max_steps` (training length) | **300,000** | derived; see Note 1 |

## Deviations from the paper, with reasoning

### Note 1: 300k steps at batch 32, vs paper's 1.2M steps at unspecified batch

The paper trains 1.2M steps on 2× A800 GPUs but does not state the per-GPU
batch size. BigCodec's stock config is `batch_size: 8` per device with
`devices: 1`, which suggests a global batch of 8 to 16 for the paper.

We pick **batch 32 on 1 H100** because:
- Empirical batch-size sweep on H100 80GB found 32 is the largest
  microbatch that fits (bs=36 OOMs at 79.16 / 79.20 GiB allocated).
  Activations dominate, not parameters.
- 300k steps at bs=32 sees 9.6M training examples, which is the same
  total as 1.2M steps at bs=8.
- Course timeline does not allow a faithful 1.2M-step reproduction at
  any batch size; the paper-faithful run on 2× A800 is 5-10 days, our
  scaled run is ~7 days on 1× H100.

### Note 2: 1× H100 80GB instead of 2× A800

The paper uses 2× A800 (data-parallel, presumably global batch ~16).
We use 1× H100 because:
- TAU's `gpu-h100-killable` partition (n-102) is what we have access to
  for this project's compute budget.
- A single H100 80GB with bs=32 sees the same per-step throughput as the
  paper's 2× A800 setup (similar generation FLOPs per step).
- Paper §3.1 confirms the recipe is single-process; DDP across 2 A800
  is just data parallelism. Reducing to 1 GPU does not change the
  optimization dynamics other than via batch size, which we account for
  via Note 3.

### Note 3: 2.0e-4 / 2.0e-5 instead of 1.0e-4 / 1.0e-5

The paper uses LR 1e-4 → 1e-5 with 1000 warmup, presumably tuned for
batch ~8-16 per the BigCodec recipe. With our batch of 32 (4x the
BigCodec default), we apply **square-root LR scaling**:

```
lr_new = lr_old * sqrt(batch_new / batch_old) = 1e-4 * sqrt(32/8) = 2e-4
```

Reasoning: linear scaling (`lr_new = 4 * lr_old = 4e-4`) is empirically
optimal for SGD on convex problems but tends to destabilize GAN training
where the effective gradient noise scale is already small at large batch
(the discriminator dampens noise via its feedback loop). Square-root
scaling is a conservative middle ground that has held up well for
adversarial recipes in practice and avoids burning a long warmup just
to recover from divergence in the first few thousand steps.

Both `gen_schedule_params` and `disc_schedule_params` are scaled
identically.

### Note 4: `down_step: 125000`, vs BigCodec's 500k

BigCodec's stock LR schedule decays linearly over 500k of its 1.2M total
steps, i.e. 41.7% of training is in the decay phase, then the remaining
58.3% holds at min_lr. We preserve the same 41.7% ratio at our reduced
step count: `125k / 300k = 41.7%`. The paper does not specify
`down_step` directly, so we follow BigCodec.

### Note 5: FSQ levels `[3,3,3,3,3,3,5,5]`

The paper says "FSQ codebook size of 18225 with projected dimension 8"
but does not list the per-dimension levels. Decomposition with 8 dims
is forced: 18225 = 3^6 × 5^2, giving six dims with 3 quantization
levels and two dims with 5 levels each. All level values are within
the FSQ paper's recommended `[3, 8]` range.

### Note 6: scaled `every_n_train_steps: 5000`

BigCodec saves the top-1 checkpoint every 10000 steps. We save the
top-3 plus `last.ckpt` every 5000 steps because (a) our shorter training
benefits from finer-grained checkpoints for analysis, and (b) `last.ckpt`
is required by our SLURM auto-resume which uses it on every requeue
across the 1-day partition boundary.

## Faithful items, for completeness

Every item in this list matches the paper exactly (or matches BigCodec
defaults where the paper is silent and BigCodec is the published
implementation the paper points to):

- LibriSpeech 960h training set
- 16 kHz 16-bit audio
- 80 Hz token frame rate
- Encoder/decoder feature dim 1024
- VQ codebook size 8192, FSQ codebook size 18225
- Projected quantizer dimension 8
- AdamW betas (0.8, 0.9)
- Linear LR decay shape, 1000 warmup steps
- Multi-resolution mel-spectrogram reconstruction loss
- HiFiGAN MPD with periods `[2, 3, 5, 7, 11]`
- Multi-STFT spec discriminator with the 5 standard FFT/hop/win
  resolutions
- Single quantizer per backbone (no residual stacking)

## Compute and SLURM

Each backbone runs as one SLURM job on TAU's `gpu-h100-killable`
partition pinned to `n-102`, with 1× H100 80GB, 8 CPUs, 96 GB RAM.
The partition limit is 24h; the SLURM script catches `TERM` 120 s
before timeout and resubmits itself, with the launcher (`auto_resume.sh`)
finding the latest `last.ckpt` and passing it as `cfg.ckpt` so
PyTorch Lightning resumes optimizer + scheduler + LR state cleanly.

Projected wall-clock at ~1.92 sec/step on 1× H100 with bs=32:
about 6.7 days, i.e. roughly 7 requeues across the 1-day partition.
W&B run IDs (`vq8k-300k`, `fsq18k-300k`) are stable across requeues so
loss curves are continuous.
