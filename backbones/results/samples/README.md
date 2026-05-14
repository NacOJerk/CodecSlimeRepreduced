# Audio samples for manual evaluation

This directory holds the first 5 utterances of UniCATS testset B, encoded
and decoded by every configuration in our 14-cell evaluation matrix.
Use it to listen to the same content under different settings and form
a perceptual judgement (the objective metrics live in
`backbones/results/final/all_metrics.csv`).

## Source utterances

All 5 are from speaker 1089, chapter 134686, of the LibriTTS test-clean
split (selected by the canonical UniCATS-B `utt2prompt` pair list,
deterministic):

| utterance ID                     | duration |
|:---------------------------------|---------:|
| 1089_134686_000002_000001        |   ~5.97 s |
| 1089_134686_000007_000005        |   ~5.31 s |
| 1089_134686_000009_000003        |   ~6.03 s |
| 1089_134686_000009_000008        |   ~5.77 s |
| 1089_134686_000015_000003        |   ~5.57 s |

Per-utterance reference text (LibriTTS `.normalized.txt`) is copied
alongside each sample as `<utt>_ref.txt`.

## Layout

```
samples/
  <variant>/
    <utt>_orig.wav     16 kHz mono, the source LibriTTS audio
    <utt>_recon.wav    same length, reconstructed by the variant
    <utt>_ref.txt      LibriTTS normalized transcript
    metrics.tsv        per-utt STOI / PESQ / SECS (no WER, no UTMOS)
    metrics_summary.json
```

## Variants (14)

Backbones (no Melt, no Cool):
- `backbone-vq8k-ffr80` - VQ-8k, no compression (80 Hz, 1040 bps)
- `backbone-vq8k-ffr40` - VQ-8k, fixed-2-frame mean to 40 Hz (520 bps)
- `backbone-vq8k-dfr40` - VQ-8k, ScheDFR DP merge to 40 Hz (600 bps)
- `backbone-fsq18k-ffr80` - FSQ-18k, no compression (80 Hz, 1132 bps)
- `backbone-fsq18k-ffr40` - FSQ-18k, fixed merge (567 bps)
- `backbone-fsq18k-dfr40` - FSQ-18k, ScheDFR (647 bps)

Melt-only (post-train with random-rate downsampling curriculum):
- `melt-fsq18k-n210-ffr40` - FSQ-18k Melt, fixed merge
- `melt-fsq18k-n210-dfr40` - FSQ-18k Melt, ScheDFR
- `melt-vq8k-n210-paperd-ffr40` - VQ-8k Melt with paper-literal d-vector, fixed
- `melt-vq8k-n210-paperd-dfr40` - VQ-8k Melt with paper-literal d-vector, ScheDFR

Cool only (encoder frozen, quantizer + decoder fine-tuned, ScheDFR active):
- `cool-vq8k-from-backbone-n210-dfr40` - VQ-8k Cool from backbone
- `cool-fsq18k-from-backbone-n210-dfr40` - FSQ-18k Cool from backbone

Melt+Cool (Cool fine-tuned on top of the Melt foundation model):
- `coolmelt-vq8k-n210-dfr40` - VQ-8k Melt then Cool
- `coolmelt-fsq18k-n210-dfr40` - FSQ-18k Melt then Cool

## Suggested listening order

For a quick A/B perception of how each adaptation stage helps, line up
the same `<utt>_recon.wav` under these four cells of the same backbone:

1. `backbone-{vq,fsq}-ffr80`        the FFR ceiling
2. `backbone-{vq,fsq}-dfr40`        ScheDFR alone, no adaptation
3. `melt-{vq,fsq}-*-dfr40`          Melt + ScheDFR
4. `coolmelt-{vq,fsq}-*-dfr40`      Melt + Cool + ScheDFR (paper headline)

## Reproducing

```
sbatch backbones/slurm/make_samples.slurm
```

Job array of 14 indices; each cell takes ~2 min on an L40s. Output goes
back into this directory (each subdir is overwritten in place).
