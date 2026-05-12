"""Evaluate COOL-only: Apply DFR compression at inference time without MELT training.

Usage:
    python evaluate_cool_only.py \\
        --ckpt path/to/last.ckpt \\
        --config codecslime_vq8k \\
        --out-dir backbones/results/cool-vq8k \\
        --down-sample-ratio 2.0 \\
        --max-compression 4

This applies DFR scheduling during inference to compress temporal redundancy.
"""
import argparse
import sys
from pathlib import Path

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from sched_dfr import SchedDFR

REPO = Path("c:/Users/noamsc/Desktop/audio_ml_tau_final-backbones-v1")
sys.path.insert(0, str(REPO / "external" / "BigCodec"))


def build_cfg(config_name: str) -> object:
    GlobalHydra.instance().clear()
    cfg_dir = REPO / "backbones" / "configs"
    bigcodec_cfg = REPO / "external" / "BigCodec" / "config"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"hydra.searchpath=[{bigcodec_cfg}]",
                f"preprocess.datasets.LibriSpeech.root={REPO}/datasets/LibriSpeech",
                "train.logger.name=cool_eval",
                "train.logger.id=cool_eval",
            ],
        )
    return cfg


def load_module(ckpt: Path, cfg) -> object:
    from lightning_module import CodecLightningModule
    monkey_cwd(REPO)
    lm = CodecLightningModule.load_from_checkpoint(str(ckpt), cfg=cfg, map_location="cuda")
    lm.eval().cuda()
    return lm


def monkey_cwd(path: Path) -> None:
    import hydra.utils
    hydra.utils.get_original_cwd = lambda: str(path)


def encode_audio(lm, wav: np.ndarray, sr: int) -> np.ndarray:
    """Encode audio to token sequence (before quantization)."""
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).cuda()
    pad = 200 - (wav_t.shape[1] % 200)
    if pad and pad != 200:
        wav_t = F.pad(wav_t, (0, pad))

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        # Get encoder output before quantization
        vq_emb = lm.model["CodecEnc"](wav_t.unsqueeze(1))
        # vq_emb is the encoded features before VQ/FSQ quantization
        return vq_emb.squeeze(0).float().cpu().numpy()


def reconstruct_from_tokens(lm, tokens: np.ndarray) -> np.ndarray:
    """Reconstruct audio from token sequence."""
    tokens_t = torch.from_numpy(tokens).float().unsqueeze(0).cuda()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        vq_post, _, _ = lm.model["generator"](tokens_t, vq=True)
        recon = lm.model["generator"](vq_post, vq=False)

    return recon.squeeze(0).squeeze(0).float().cpu().numpy()


def apply_cool_compression(tokens: np.ndarray, dfr: SchedDFR) -> tuple:
    """Apply COOL (DFR) compression to token sequence."""
    # Apply optimal downsampling
    encoded = dfr.optimal_down_sample(tokens)

    # Calculate compression statistics
    original_frames = len(tokens)
    compressed_frames = len(encoded.encoded_data)
    compression_ratio = original_frames / compressed_frames

    # Upsample back to original length for reconstruction
    reconstructed_tokens = dfr.up_sample_encoded(encoded)

    return reconstructed_tokens, compression_ratio, encoded.encoding_lengths


def calculate_metrics(orig_wav: np.ndarray, recon_wav: np.ndarray, sr: int) -> dict:
    """Calculate basic audio quality metrics."""
    # Ensure same length
    min_len = min(len(orig_wav), len(recon_wav))
    orig_wav = orig_wav[:min_len]
    recon_wav = recon_wav[:min_len]

    # Simple metrics
    mse = np.mean((orig_wav - recon_wav) ** 2)
    rmse = np.sqrt(mse)
    snr = 10 * np.log10(np.mean(orig_wav ** 2) / mse) if mse > 0 else float('inf')

    return {
        'mse': mse,
        'rmse': rmse,
        'snr_db': snr,
        'duration_s': min_len / sr
    }


def pick_samples(filelist: Path, n: int, seed: int) -> list:
    lines = [ln.strip().split("|") for ln in filelist.read_text().splitlines() if ln.strip()]
    if n >= len(lines):
        return lines
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(lines), size=n, replace=False)
    return [lines[i] for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, choices=["codecslime_vq8k", "codecslime_fsq18k"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--down-sample-ratio", type=float, default=2.0,
                   help="Target downsampling ratio for DFR")
    ap.add_argument("--max-compression", type=int, default=4,
                   help="Maximum compression factor")
    ap.add_argument("--n-samples", type=int, default=10,
                   help="Number of samples to evaluate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DFR scheduler
    dfr = SchedDFR(down_sample_ratio=args.down_sample_ratio,
                   max_compression=args.max_compression)

    cfg = build_cfg(args.config)
    lm = load_module(args.ckpt, cfg)

    sr = cfg.preprocess.audio.sr
    root = REPO / "datasets" / "LibriSpeech"
    dev_filelist = REPO / "data" / "librispeech_dev.txt"

    if not dev_filelist.exists():
        print(f"Error: Dev filelist not found at {dev_filelist}")
        return

    # Pick samples
    chosen = pick_samples(dev_filelist, args.n_samples, args.seed)

    results = []
    print(f"Evaluating {len(chosen)} samples with COOL (DFR ratio={args.down_sample_ratio}, max_comp={args.max_compression})")

    for fid, relpath in tqdm(chosen):
        wavpath = root / relpath
        if not wavpath.exists():
            print(f"Warning: {wavpath} not found, skipping")
            continue

        # Load original audio
        wav, _ = librosa.load(wavpath, sr=sr)

        # Step 1: Standard reconstruction (no compression)
        tokens_full = encode_audio(lm, wav, sr)
        recon_full = reconstruct_from_tokens(lm, tokens_full)

        # Step 2: COOL reconstruction (with DFR compression)
        tokens_compressed, comp_ratio, encoding_lengths = apply_cool_compression(tokens_full, dfr)
        recon_compressed = reconstruct_from_tokens(lm, tokens_compressed)

        # Calculate metrics
        metrics_full = calculate_metrics(wav, recon_full, sr)
        metrics_compressed = calculate_metrics(wav, recon_compressed, sr)

        # Save audio files
        orig_path = args.out_dir / f"{fid}_orig.wav"
        full_recon_path = args.out_dir / f"{fid}_full_recon.wav"
        cool_recon_path = args.out_dir / f"{fid}_cool_recon.wav"

        sf.write(orig_path, wav, sr)
        sf.write(full_recon_path, recon_full, sr)
        sf.write(cool_recon_path, recon_compressed, sr)

        # Store results
        result = {
            'fid': fid,
            'compression_ratio': comp_ratio,
            'encoding_lengths': encoding_lengths,
            'metrics_full': metrics_full,
            'metrics_compressed': metrics_compressed,
            'duration_s': metrics_full['duration_s']
        }
        results.append(result)

        print(f"  {fid}: ratio={comp_ratio:.2f}, full_SNR={metrics_full['snr_db']:.1f}dB, "
              f"cool_SNR={metrics_compressed['snr_db']:.1f}dB")

    # Save summary
    summary_path = args.out_dir / "cool_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"COOL Evaluation Summary\n")
        f.write(f"Model: {args.config}\n")
        f.write(f"DFR ratio: {args.down_sample_ratio}\n")
        f.write(f"Max compression: {args.max_compression}\n")
        f.write(f"Samples: {len(results)}\n\n")

        avg_comp_ratio = np.mean([r['compression_ratio'] for r in results])
        avg_snr_full = np.mean([r['metrics_full']['snr_db'] for r in results])
        avg_snr_compressed = np.mean([r['metrics_compressed']['snr_db'] for r in results])

        f.write(f"Average compression ratio: {avg_comp_ratio:.2f}\n")
        f.write(f"Average SNR (full): {avg_snr_full:.2f} dB\n")
        f.write(f"Average SNR (compressed): {avg_snr_compressed:.2f} dB\n")
        f.write(f"SNR degradation: {avg_snr_full - avg_snr_compressed:.2f} dB\n\n")

        f.write("Per-sample results:\n")
        for r in results:
            f.write(f"{r['fid']}: ratio={r['compression_ratio']:.2f}, "
                   f"full_SNR={r['metrics_full']['snr_db']:.1f}dB, "
                   f"cool_SNR={r['metrics_compressed']['snr_db']:.1f}dB\n")

    print(f"\nDone. Results saved to {args.out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()