"""End-to-end evaluation of a codec checkpoint (FFR or ScheDFR mode).

Reads a manifest of `fid|relpath` rows, encodes/decodes each utterance with
the chosen mode, writes paired orig/recon wavs (plus a `_ref.txt` when a
LibriTTS-style `.normalized.txt` neighbour exists), then computes WER, STOI,
PESQ, SECS, UTMOS in-process via `eval_metrics.compute_*`.

Usage (smoke, 5 utts):

  python backbones/scripts/evaluate_codec.py \
      --ckpt backbones/checkpoints/vq8k-300k/last.ckpt \
      --manifest backbones/data/unicats_b.txt \
      --audio-root datasets/LibriTTS \
      --mode dfr \
      --out-dir backbones/results/smoke-eval-vq8k-dfr \
      --codebook-size 8192 \
      --limit 5 --whisper-model tiny
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backbones" / "scripts"))

from _codec_loader import load_codec  # noqa: E402
from sched_dfr import SchedDFR  # noqa: E402
import eval_metrics  # noqa: E402

SR = 16000
ENCODER_HOP = 200  # encoder downsample factor; 16000 / 200 = 80 Hz


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        fid, relpath = ln.split("|", 1)
        rows.append((fid, relpath))
    return rows


def _load_reference_text(wav_path: Path, suffix: str) -> str | None:
    if not suffix:
        return None
    ref_path = wav_path.with_suffix(suffix)
    if ref_path.exists():
        return ref_path.read_text().strip()
    return None


def _encode(encoder, wav: np.ndarray, device: str) -> torch.Tensor:
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    pad = (ENCODER_HOP - wav_t.shape[1] % ENCODER_HOP) % ENCODER_HOP
    if pad:
        wav_t = F.pad(wav_t, (0, pad))
    wav_t = wav_t.to(device)
    return encoder(wav_t.unsqueeze(1))


def _decode_ffr(decoder, vq_emb: torch.Tensor) -> np.ndarray:
    vq_post, _, _ = decoder(vq_emb, vq=True)
    out = decoder(vq_post, vq=False).squeeze(0).squeeze(0).cpu().numpy()
    return out


def _decode_dfr(decoder, vq_emb: torch.Tensor, dfr: SchedDFR, device: str) -> tuple[np.ndarray, float, list[int]]:
    tokens = vq_emb.squeeze(0).T.detach().cpu().float().numpy()  # [T_enc, C]
    encoded = dfr.optimal_down_sample(tokens)
    tokens_comp = dfr.up_sample_encoded(encoded)
    comp_ratio = len(tokens) / len(encoded.encoded_data)
    t_comp = torch.from_numpy(tokens_comp.T).float().unsqueeze(0).to(device)
    vq_post, _, _ = decoder(t_comp, vq=True)
    out = decoder(vq_post, vq=False).squeeze(0).squeeze(0).cpu().numpy()
    return out, comp_ratio, list(encoded.encoding_lengths)


def _decode_fixed(decoder, vq_emb: torch.Tensor, rs: float, device: str) -> tuple[np.ndarray, float, list[int]]:
    # Paper's "fixed-rate merge" baseline: split encoder frames into uniform
    # groups of size `rs` and replace each by its mean. Same mean-broadcast
    # apply as ScheDFR, but with no DP search over segment lengths.
    s = int(round(rs))
    tokens = vq_emb.squeeze(0).T.detach().cpu().float().numpy()  # [T_enc, C]
    T_enc = tokens.shape[0]
    n_full = T_enc // s
    leftover = T_enc - n_full * s
    encoding_lengths = [s] * n_full + ([leftover] if leftover else [])
    downsampled = SchedDFR.down_sample(tokens, encoding_lengths)
    replicated = SchedDFR.up_sample(downsampled, encoding_lengths)
    comp_ratio = T_enc / len(downsampled)
    t_comp = torch.from_numpy(replicated.T).float().unsqueeze(0).to(device)
    vq_post, _, _ = decoder(t_comp, vq=True)
    out = decoder(vq_post, vq=False).squeeze(0).squeeze(0).cpu().numpy()
    return out, comp_ratio, encoding_lengths


def _try_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception as exc:
        return f"ERR({type(exc).__name__}: {exc})"


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), float)]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--audio-root", required=True, type=Path,
                    help="parent dir of `relpath` (e.g. datasets/LibriTTS or datasets/LibriSpeech)")
    ap.add_argument("--mode", required=True, choices=["ffr", "dfr", "fixed"],
                    help="ffr=no downsample (80Hz), fixed=uniform Rs-frame mean (40Hz), "
                         "dfr=ScheDFR DP (40Hz)")
    ap.add_argument("--rs", type=float, default=2.0, help="DFR/fixed down-sample ratio")
    ap.add_argument("--u", type=int, default=4, help="DFR max compression length")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--codebook-size", type=int, required=True,
                    help="8192 for vq8k, 18225 for fsq18k (used for bitrate)")
    ap.add_argument("--encoder-fr", type=float, default=80.0,
                    help="encoder native frame rate (Hz); BigCodec = 80 at 16 kHz")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--ref-text-suffix", default=".normalized.txt",
                    help="suffix replacing .wav to locate reference text; empty disables")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all manifest entries; otherwise first N")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-wer", action="store_true")
    ap.add_argument("--no-utmos", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha1 = hashlib.sha1(manifest_bytes).hexdigest()
    rows_manifest = _read_manifest(args.manifest)
    if args.limit > 0:
        rows_manifest = rows_manifest[:args.limit]
    print(f"[manifest] {args.manifest.name} sha1={manifest_sha1[:10]} rows={len(rows_manifest)}")

    encoder, decoder, model_name = load_codec(args.ckpt, args.device)
    print(f"[codec] {model_name} ckpt={args.ckpt.name} device={args.device}")

    dfr = SchedDFR(down_sample_ratio=args.rs, max_compression=args.u) if args.mode == "dfr" else None
    if args.mode == "fixed" and abs(args.rs - round(args.rs)) > 1e-6:
        raise SystemExit(f"--mode fixed requires integer --rs (got {args.rs})")

    whisper_model = None
    use_wer = not args.no_wer and eval_metrics._whisper is not None and eval_metrics._jiwer is not None
    if use_wer:
        print(f"[wer] loading Whisper '{args.whisper_model}'")
        whisper_model = eval_metrics._whisper.load_model(args.whisper_model)

    rows: list[dict] = []
    for fid, relpath in tqdm(rows_manifest, desc="eval"):
        wav_path = args.audio_root / relpath
        if not wav_path.exists():
            print(f"[skip] {wav_path} not found")
            continue
        wav, _ = librosa.load(str(wav_path), sr=SR)

        ref_text = _load_reference_text(wav_path, args.ref_text_suffix)

        with torch.no_grad():
            vq_emb = _encode(encoder, wav, args.device)
            if args.mode == "ffr":
                recon = _decode_ffr(decoder, vq_emb)
                comp_ratio = 1.0
                encoding_lengths: list[int] = []
            elif args.mode == "fixed":
                recon, comp_ratio, encoding_lengths = _decode_fixed(decoder, vq_emb, args.rs, args.device)
            else:
                recon, comp_ratio, encoding_lengths = _decode_dfr(decoder, vq_emb, dfr, args.device)

        recon = recon[:len(wav)]

        sf.write(str(args.out_dir / f"{fid}_orig.wav"), wav, SR)
        sf.write(str(args.out_dir / f"{fid}_recon.wav"), recon, SR)
        if ref_text is not None:
            (args.out_dir / f"{fid}_ref.txt").write_text(ref_text + "\n")

        row: dict = {
            "fid": fid,
            "duration_s": float(len(wav) / SR),
            "comp_ratio": float(comp_ratio),
        }
        if args.mode == "dfr":
            row["encoding_lengths"] = encoding_lengths

        if use_wer:
            row["wer"] = _try_metric(eval_metrics.compute_wer, wav, recon, whisper_model,
                                     reference_text=ref_text)
        else:
            row["wer"] = None

        row["stoi"] = _try_metric(eval_metrics.compute_stoi, wav, recon, SR) \
            if eval_metrics._pystoi is not None else None
        row["pesq"] = _try_metric(eval_metrics.compute_pesq, wav, recon, SR) \
            if eval_metrics._pesq_mod is not None else None
        row["secs"] = _try_metric(eval_metrics.compute_secs, wav, recon, SR) \
            if eval_metrics._resemblyzer is not None else None
        row["utmos"] = _try_metric(eval_metrics.compute_utmos, recon, SR) \
            if not args.no_utmos else None

        rows.append(row)

    # Duration bits are needed only for ScheDFR; the fixed-merge schedule is
    # known a priori, and FFR has no per-frame schedule at all.
    duration_bits = math.ceil(math.log2(args.u)) if args.mode == "dfr" else 0
    mean_comp_ratio = float(np.mean([r["comp_ratio"] for r in rows])) if rows else 1.0
    bits_per_frame = math.log2(args.codebook_size) + duration_bits
    bitrate_bps = bits_per_frame * args.encoder_fr / mean_comp_ratio

    averages = {
        "wer": _avg(rows, "wer"),
        "stoi": _avg(rows, "stoi"),
        "pesq": _avg(rows, "pesq"),
        "secs": _avg(rows, "secs"),
        "utmos": _avg(rows, "utmos"),
        "comp_ratio": mean_comp_ratio,
        "bitrate_bps": bitrate_bps,
        "duration_bits_per_frame": duration_bits,
        "n_utt": len(rows),
    }

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "manifest_sha1": manifest_sha1,
        "model_name": model_name,
        "averages": averages,
        "per_utt": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["fid", "duration_s", "comp_ratio", "wer", "stoi", "pesq", "secs", "utmos"]
    with open(args.out_dir / "metrics.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print()
    print("=" * 70)
    print(f"AVERAGES  (n={averages['n_utt']}, mode={args.mode}, model={model_name})")
    print(f"  bitrate {bitrate_bps:8.1f} bps "
          f"(= [log2({args.codebook_size}) + {duration_bits}] * {args.encoder_fr} / {mean_comp_ratio:.3f})")
    for k in ("wer", "stoi", "pesq", "secs", "utmos"):
        v = averages[k]
        arrow = "down" if k == "wer" else "up "
        print(f"  {k.upper():<5} {arrow} {v:.4f}" if v is not None else f"  {k.upper():<5}     N/A")
    print("=" * 70)


if __name__ == "__main__":
    main()
