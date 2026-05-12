"""Compute speech quality metrics between paired original/reconstructed audio.

Metrics: WER↓  STOI↑  PESQ↑  SECS↑  ViSQOL↑  UTMOS↑

Usage:
    # When eval scripts write both *_orig.wav and *_recon.wav to the same dir:
    python eval_metrics.py --dir backbones/results/vanilla-fsq18k

    # Or with explicit separate dirs / suffixes:
    python eval_metrics.py --orig-dir path/to/orig --recon-dir path/to/recon
    python eval_metrics.py --dir results/ --orig-suffix _orig --recon-suffix _full_recon

Install missing metric packages:
    pip install openai-whisper jiwer          # WER
    pip install pystoi                        # STOI
    pip install pesq                          # PESQ
    pip install resemblyzer                   # SECS
    pip install visqol                        # ViSQOL (optional, needs extra C++ deps)
"""
import argparse
import math
from pathlib import Path

import librosa
import numpy as np

REPO = Path(__file__).parent.parent.parent

# ── optional metric backends ──────────────────────────────────────────────────

def _try_import(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        return None


# WER
_whisper = _try_import("whisper")
_jiwer = _try_import("jiwer")

# STOI
_pystoi = _try_import("pystoi")

# PESQ
_pesq_mod = _try_import("pesq")

# SECS – encoder loaded once and reused
_resemblyzer = _try_import("resemblyzer")
_voice_encoder = None

# UTMOS – loaded lazily on first use via torch.hub
_utmos_predictor = None

# ── metric functions ──────────────────────────────────────────────────────────

def compute_wer(orig: np.ndarray, recon: np.ndarray, model) -> float:
    # pass float32 numpy arrays at 16 kHz — avoids ffmpeg dependency
    orig_text  = model.transcribe(orig.astype(np.float32))["text"].strip().lower()
    recon_text = model.transcribe(recon.astype(np.float32))["text"].strip().lower()
    import jiwer
    return jiwer.wer(orig_text, recon_text)


def compute_stoi(orig: np.ndarray, recon: np.ndarray, sr: int) -> float:
    from pystoi import stoi
    min_len = min(len(orig), len(recon))
    return stoi(orig[:min_len], recon[:min_len], sr, extended=False)


def compute_pesq(orig: np.ndarray, recon: np.ndarray, sr: int) -> float:
    from pesq import pesq
    min_len = min(len(orig), len(recon))
    mode = "wb" if sr >= 16000 else "nb"
    target_sr = 16000 if mode == "wb" else 8000
    if sr != target_sr:
        orig = librosa.resample(orig, orig_sr=sr, target_sr=target_sr)
        recon = librosa.resample(recon, orig_sr=sr, target_sr=target_sr)
        min_len = min(len(orig), len(recon))
    return pesq(target_sr, orig[:min_len], recon[:min_len], mode)


def compute_secs(orig: np.ndarray, recon: np.ndarray, sr: int) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav
    global _voice_encoder
    if _voice_encoder is None:
        _voice_encoder = VoiceEncoder()
    if sr != 16000:
        orig = librosa.resample(orig, orig_sr=sr, target_sr=16000)
        recon = librosa.resample(recon, orig_sr=sr, target_sr=16000)
    emb1 = _voice_encoder.embed_utterance(preprocess_wav(orig, source_sr=16000))
    emb2 = _voice_encoder.embed_utterance(preprocess_wav(recon, source_sr=16000))
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))



def compute_utmos(recon: np.ndarray, sr: int) -> float:
    import torch
    global _utmos_predictor
    if _utmos_predictor is None:
        _utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        _utmos_predictor.eval()
    wav_t = torch.from_numpy(recon).float().unsqueeze(0)
    if sr != 16000:
        import torchaudio.functional as TAF
        wav_t = TAF.resample(wav_t, sr, 16000)
    with torch.no_grad():
        score = _utmos_predictor(wav_t, 16000)
    return float(score.mean())


# ── file pairing ──────────────────────────────────────────────────────────────

def find_pairs(orig_dir: Path, recon_dir: Path, orig_suffix: str, recon_suffix: str):
    orig_files = {
        f.name[: -len(f"{orig_suffix}.wav")]: f
        for f in sorted(orig_dir.glob(f"*{orig_suffix}.wav"))
    }
    recon_files = {
        f.name[: -len(f"{recon_suffix}.wav")]: f
        for f in sorted(recon_dir.glob(f"*{recon_suffix}.wav"))
    }
    common = sorted(set(orig_files) & set(recon_files))
    if not common:
        raise ValueError(
            f"No matching pairs found.\n"
            f"  orig_dir ({orig_dir}): {len(orig_files)} files with suffix '{orig_suffix}'\n"
            f"  recon_dir ({recon_dir}): {len(recon_files)} files with suffix '{recon_suffix}'"
        )
    return [(common[i], orig_files[common[i]], recon_files[common[i]]) for i in range(len(common))]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None,
                    help="Single directory containing both orig and recon files")
    ap.add_argument("--orig-dir", type=Path, default=None)
    ap.add_argument("--recon-dir", type=Path, default=None)
    ap.add_argument("--orig-suffix", type=str, default="_orig")
    ap.add_argument("--recon-suffix", type=str, default="_recon")
    ap.add_argument("--out", type=Path, default=None, help="Save results to this txt file")
    ap.add_argument("--whisper-model", type=str, default="base",
                    help="Whisper model size for WER (tiny/base/small/medium/large)")
    ap.add_argument("--no-wer", action="store_true", help="Skip WER (slow)")
    ap.add_argument("--no-utmos", action="store_true", help="Skip UTMOS (downloads model)")
    # Bitrate
    ap.add_argument("--codebook-size", type=int, default=None,
                    help="Codec codebook size (18225 for fsq18k, 8192 for vq8k). "
                         "Omit to skip bitrate.")
    ap.add_argument("--frame-rate", type=float, default=80.0,
                    help="Codec frame rate in Hz (default 80 = 16000/200)")
    ap.add_argument("--compression-ratio", type=float, default=1.0,
                    help="DFR compression ratio for COOL evaluation (default 1.0 = no compression)")
    args = ap.parse_args()

    if args.dir is not None:
        orig_dir = recon_dir = args.dir
    else:
        if args.orig_dir is None or args.recon_dir is None:
            ap.error("Provide either --dir or both --orig-dir and --recon-dir")
        orig_dir, recon_dir = args.orig_dir, args.recon_dir

    pairs = find_pairs(orig_dir, recon_dir, args.orig_suffix, args.recon_suffix)
    print(f"Found {len(pairs)} pairs  (suffix: '{args.orig_suffix}' ↔ '{args.recon_suffix}')\n")

    # ── availability check ────────────────────────────────────────────────────
    use_wer   = not args.no_wer and _whisper is not None and _jiwer is not None
    use_stoi  = _pystoi is not None
    use_pesq  = _pesq_mod is not None
    use_secs  = _resemblyzer is not None
    use_utmos = not args.no_utmos  # always try via torch.hub

    missing = []
    if not use_wer:  missing.append("WER  → pip install openai-whisper jiwer")
    if not use_stoi: missing.append("STOI → pip install pystoi")
    if not use_pesq: missing.append("PESQ → pip install pesq")
    if not use_secs: missing.append("SECS → pip install resemblyzer")
    if missing:
        print("Missing packages (metrics will show N/A):")
        for m in missing:
            print(f"  {m}")
        print()

    # ── bitrate (constant across all files for a given codec/setting) ─────────
    if args.codebook_size is not None:
        bits_per_frame = math.log2(args.codebook_size)
        bitrate_bps = bits_per_frame * args.frame_rate / args.compression_ratio
        bitrate_kbps = bitrate_bps / 1000
        print(f"Bitrate: log2({args.codebook_size}) × {args.frame_rate} Hz "
              f"/ {args.compression_ratio} = {bitrate_bps:.1f} bps  ({bitrate_kbps:.3f} kbps)\n")
    else:
        bitrate_kbps = None

    # ── load Whisper once ─────────────────────────────────────────────────────
    whisper_model = None
    if use_wer:
        print(f"Loading Whisper ({args.whisper_model}) for WER…")
        whisper_model = _whisper.load_model(args.whisper_model)

    # ── per-file computation ──────────────────────────────────────────────────
    rows = []
    for fid, orig_path, recon_path in pairs:
        sr = 16000
        orig,  _ = librosa.load(str(orig_path),  sr=sr)
        recon, _ = librosa.load(str(recon_path), sr=sr)

        row = {"fid": fid}

        if use_wer:
            try:
                row["wer"] = compute_wer(orig, recon, whisper_model)
            except Exception as e:
                row["wer"] = f"ERR({e})"
        else:
            row["wer"] = None

        if use_stoi:
            try:
                row["stoi"] = compute_stoi(orig, recon, sr)
            except Exception as e:
                row["stoi"] = f"ERR({e})"
        else:
            row["stoi"] = None

        if use_pesq:
            try:
                row["pesq"] = compute_pesq(orig, recon, sr)
            except Exception as e:
                row["pesq"] = f"ERR({e})"
        else:
            row["pesq"] = None

        if use_secs:
            try:
                row["secs"] = compute_secs(orig, recon, sr)
            except Exception as e:
                row["secs"] = f"ERR({e})"
        else:
            row["secs"] = None

        if use_utmos:
            try:
                row["utmos"] = compute_utmos(recon, sr)
            except Exception as e:
                row["utmos"] = f"ERR({e})"
        else:
            row["utmos"] = None

        rows.append(row)

        def _fmt(v):
            if v is None:       return "  N/A  "
            if isinstance(v, float): return f"{v:7.4f}"
            return f"  {v}  "

        print(f"  {fid:30s}  WER={_fmt(row['wer'])}  STOI={_fmt(row['stoi'])}  "
              f"PESQ={_fmt(row['pesq'])}  SECS={_fmt(row['secs'])}  UTMOS={_fmt(row['utmos'])}")

    # ── averages ──────────────────────────────────────────────────────────────
    def _avg(key):
        vals = [r[key] for r in rows if isinstance(r[key], float)]
        return np.mean(vals) if vals else None

    avgs = {k: _avg(k) for k in ["wer", "stoi", "pesq", "secs", "utmos"]}

    lines = [
        "",
        "=" * 70,
        "AVERAGES",
        f"  Bitrate  {bitrate_kbps:.3f} kbps" if bitrate_kbps is not None else "  Bitrate  N/A  (pass --codebook-size)",
        f"  WER↓    {avgs['wer']  :.4f}" if avgs["wer"]   is not None else "  WER↓    N/A",
        f"  STOI↑   {avgs['stoi'] :.4f}" if avgs["stoi"]  is not None else "  STOI↑   N/A",
        f"  PESQ↑   {avgs['pesq'] :.4f}" if avgs["pesq"]  is not None else "  PESQ↑   N/A",
        f"  SECS↑   {avgs['secs'] :.4f}" if avgs["secs"]  is not None else "  SECS↑   N/A",
        f"  UTMOS↑  {avgs['utmos']:.4f}" if avgs["utmos"] is not None else "  UTMOS↑  N/A",
        "=" * 70,
    ]
    print("\n".join(lines))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in rows:
                f.write(f"{r['fid']}\t"
                        f"wer={r['wer']}\tstoi={r['stoi']}\tpesq={r['pesq']}\t"
                        f"secs={r['secs']}\tutmos={r['utmos']}\n")
            f.write("\nAVERAGES\n")
            if bitrate_kbps is not None:
                f.write(f"bitrate={bitrate_kbps:.3f} kbps\n")
            for k, v in avgs.items():
                f.write(f"{k}={v:.4f}\n" if v is not None else f"{k}=N/A\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()