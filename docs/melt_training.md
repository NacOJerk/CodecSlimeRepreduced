# Melt-stage training: hyperparameter decisions

This doc records the configuration choices for the Melt post-training stage of
CodecSlime, as we implement it on top of the BigCodec backbones we trained in
Stage 1 (`backbones/checkpoints/{vq8k,fsq18k}-300k/last.ckpt`).

## Step budget

100k steps, matching the paper. This also matches the default `s_p=1e5` already
in `melt_manager.py`, so the curriculum hits `p_tgt` exactly at the end of the
schedule.

## Learning rate

This is the only nontrivial choice. The paper is explicit but its setup differs
from ours, so we record the evidence, the options, and the decision.

### What the paper says

`papers/codecslime_2506.21074_source/main_icassp26.tex` §3.1 (Setup):

> "We use AdamW optimizer with moving average coefficients beta_1=0.8 and beta_2=0.9,
> and a linearly decaying learning rate from 10^-4 to 10^-5 with 1000 warm-up
> steps. The backbone model is trained for 1.2M steps."
> "The Melt stage still uses the same learning rate configuration as the backbone."
> "In the Cool stage [...] the learning rate decays from 4e-5 to 1e-5."

The project page <https://x-lance.github.io/codecslime/> contains the Melt
algorithm box but adds no LR detail beyond what is in the paper.

So the paper's prescription for Melt is: AdamW, betas (0.8, 0.9), warmup 1000,
linear decay from 1e-4 to 1e-5. Applied over Melt's 100k step window, that means
re-warm from near-zero up to 1e-4 over 1000 steps, then linearly down to 1e-5
across the remaining 99k.

That Cool uses a lower LR (4e-5 -> 1e-5) and Melt uses the higher backbone schedule
is consistent with the idea that Melt is a heavier post-training pass and Cool
is a lighter fine-tune.

### Our backbone is not on the paper's scale

Our backbone was trained with `max_lr=2e-4, min_lr=2e-5, warmup_step=1000,
down_step=125000` (config `backbones/configs/train/codecslime_300k.yaml`), with
`WarmupLR` from `external/BigCodec/common/schedulers.py`:

- steps 0 to 1000: quadratic warmup to `max_lr=2e-4`
- steps 1000 to 126000: linear decay from `max_lr=2e-4` to `min_lr=2e-5`
- steps 126000+: constant at `min_lr=2e-5`

At step ~245k (current `last.ckpt`), our backbone is at `2e-5`, i.e. 2x the
paper's final LR. The 2x came from BigCodec's defaults; we did not explicitly
revisit it for the paper match.

### Options considered

| Option | Schedule | Pros | Cons |
|---|---|---|---|
| A. Paper-literal | warmup=1000, max=1e-4, min=1e-5, decay over 99k | Faithful to the paper's "same LR config as backbone". Bounded below our backbone's training max. | Re-warming a converged model to 1e-4 (5x its current LR) can destabilize discriminator/quantizer. |
| B. Our-backbone-literal | warmup=1000, max=2e-4, min=2e-5, decay over 99k | "Same config as *our* backbone." | Re-warming to 2e-4 on a converged model is the most aggressive of all options. |
| C. Light fine-tune re-warm | warmup=500, max=5e-5, min=1e-5, decay over 99k | Geometric mid between current 2e-5 and the paper's 1e-4. Gives headroom without resetting to peak. | Not paper-literal. |
| D. No re-warm, low constant | constant 2e-5 (or 1e-5) for all 100k | Safest. No training instability. | Likely under-fits the new compression task; defeats the point of Melt. |

### Decision

**Primary: Option A (paper-literal re-warm, 1e-4 -> 1e-5, 1000 warmup, linear decay over 99k).**

Reasoning:
1. The paper is unambiguous about which schedule it uses for Melt.
2. Our backbone trained at *2x* the paper's max LR, so warming up to *1e-4* is
   actually below our backbone's training maximum. The model has already
   tolerated 2e-4 once.
3. 1000 warmup steps absorb the initial instability of re-engaging the optimizer
   on a converged model.

**Fallback: Option C (5e-5 -> 1e-5, warmup 500) if Option A produces instability.**

Indicators of instability we will watch in the first ~5k steps of the smoke run:
- `mel_loss` blowing up above ~5x its post-backbone value.
- `disc_loss` collapsing toward 0 (discriminator wins, generator can't recover).
- `adv_loss` diverging (generator collapses).
- NaN in `vq_loss` or codebook usage falling off a cliff (VQ-8k only).

If any of these appear within 5k steps, we restart with Option C from the same
backbone checkpoint and document the switch.

### `WarmupLR` parameter values to use

For Option A, the four parameters consumed by `WarmupLR`:

```yaml
gen_schedule_params:
  warmup_step: 1000
  down_step: 99000     # 100k total - 1000 warmup
  min_lr: 1.0e-5
  max_lr: 1.0e-4
disc_schedule_params:
  warmup_step: 1000
  down_step: 99000
  min_lr: 1.0e-5
  max_lr: 1.0e-4
```

The two optimizers share the schedule, matching backbone behavior.

## Other hyperparameters

These follow the paper / our backbone defaults with no special reasoning needed:

- Optimizer: AdamW, betas (0.8, 0.9), same as backbone.
- Batch: same as backbone (LibriSpeech 1-second crops, batch size from
  `backbones/configs/dataset/librispeech.yaml`).
- Mixed precision: `16-mixed`, same as backbone.
- Gradient clipping: norm 1.0 for both gen and disc, same as backbone.
- Loss weights: `lambda_mel=15.0, lambda_adv=1.0, lambda_fm=1.0, lambda_disc=1.0`,
  same as backbone (which itself follows BigCodec defaults).
- Checkpoint: every 5000 steps, `save_top_k=3`, `save_last=true`.
- Discriminators: kept trainable throughout Melt. Paper is silent on freezing
  any module during Melt; only Cool freezes the encoder.
- All four sub-modules (`CodecEnc`, `generator`, `discriminator`,
  `spec_discriminator`) are trainable during Melt.

## Melt-manager hyperparameters

Defaults already in `melt_manager.py`, matching the project page:

- `max_compression = 4`
- `p_tgt = [0.1, 0.45, 0.25, 0.2]`
- `s_p = 100000` (matches our 100k step budget)
- `concentration_control = 30.0`
- `skip_prob = 0.5`
- `USE_PAPER_D_ENFORCE = False` (deliberate; see CLAUDE.md and the plan's audit section)

## Backbone starting points

We Melt-train both backbones in sequence:

1. VQ-8k: `backbones/checkpoints/vq8k-300k/last.ckpt` (currently step ~245k of a
   planned 300k backbone schedule).
2. FSQ-18k: `backbones/checkpoints/fsq18k-300k/last.ckpt` (same).

Neither backbone reached its planned 300k step budget. The user has opted to
proceed from the current `last.ckpt` rather than wait. Worth a one-line caveat
in the project PDF.

## Things we deliberately did *not* tune

- `p_tgt`: literal paper / page values. Changing this changes the target frame-rate
  distribution and would invalidate downstream Cool comparisons.
- `concentration_control`: literal page value.
- `skip_prob`: literal page value.
- `s_p`: tied to step budget by construction.
- Loss weights: unchanged from backbone training.

If any of these need to change after the run, document the change here with the
reason.
