"""COOL evaluation: apply DFR compression at inference time without retraining."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from tqdm import tqdm

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "BigCodec"))

from sched_dfr import SchedDFR
from omegaconf import OmegaConf
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder


def load_codec(ckpt_path: Path, device: str):
    ckpt_name = ckpt_path.name
    is_fsq = "fsq" in ckpt_name
    model_name = "fsq18k" if is_fsq else "vq8k"
    cfg = OmegaConf.load(REPO / "backbones" / "configs" / "model" / f"{model_name}.yaml")

    enccfg = cfg.codec_encoder
    encoder = CodecEncoder(
        ngf=enccfg.ngf,
        use_rnn=enccfg.use_rnn,
        rnn_bidirectional=enccfg.rnn_bidirectional,
        rnn_num_layers=enccfg.rnn_num_layers,
        up_ratios=list(enccfg.up_ratios),
        dilations=list(enccfg.dilations),
        out_channels=enccfg.out_channels,
    )

    deccfg = cfg.codec_decoder
    quantizer_type = deccfg.get("quantizer_type", "vq")
    dec_kwargs = dict(
        in_channels=deccfg.in_channels,
        upsample_initial_channel=deccfg.upsample_initial_channel,
        ngf=deccfg.ngf,
        use_rnn=deccfg.use_rnn,
        rnn_bidirectional=deccfg.rnn_bidirectional,
        rnn_num_layers=deccfg.rnn_num_layers,
        up_ratios=list(deccfg.up_ratios),
        dilations=list(deccfg.dilations),
        vq_dim=deccfg.vq_dim,
        quantizer_type=quantizer_type,
    )
    if quantizer_type == "fsq":
        dec_kwargs.update(fsq_levels=list(deccfg.fsq_levels), fsq_dim=deccfg.fsq_dim)
    else:
        dec_kwargs.update(
            vq_num_quantizers=deccfg.vq_num_quantizers,
            vq_commit_weight=deccfg.vq_commit_weight,
            vq_full_commit_loss=deccfg.vq_full_commit_loss,
            codebook_size=deccfg.codebook_size,
            codebook_dim=deccfg.codebook_dim,
        )
    decoder = CodecDecoder(**dec_kwargs)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("model.CodecEnc."):]: v for k, v in sd.items() if k.startswith("model.CodecEnc.")}
    dec_sd = {k[len("model.generator."):]: v for k, v in sd.items() if k.startswith("model.generator.")}
    encoder.load_state_dict(enc_sd)
    decoder.load_state_dict(dec_sd)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    print(f"Loaded {len(enc_sd)} encoder params, {len(dec_sd)} decoder params")
    return encoder, decoder


def calculate_metrics(orig_wav: np.ndarray, recon_wav: np.ndarray, sr: int) -> dict:
    min_len = min(len(orig_wav), len(recon_wav))
    orig_wav = orig_wav[:min_len]
    recon_wav = recon_wav[:min_len]
    mse = np.mean((orig_wav - recon_wav) ** 2)
    snr = 10 * np.log10(np.mean(orig_wav ** 2) / mse) if mse > 0 else float("inf")
    return {"mse": float(mse), "snr_db": float(snr), "duration_s": min_len / sr}


def pick_samples(filelist: Path, n: int, seed: int) -> list:
    lines = [ln.strip().split("|") for ln in filelist.read_text().splitlines() if ln.strip()]
    if n >= len(lines):
        return lines
    rng = np.random.default_rng(seed)
    return [lines[i] for i in rng.choice(len(lines), size=n, replace=False)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--down-sample-ratio", type=float, default=2.0)
    ap.add_argument("--max-compression", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dfr = SchedDFR(down_sample_ratio=args.down_sample_ratio, max_compression=args.max_compression)
    encoder, decoder = load_codec(args.ckpt, args.device)

    sr = 16000
    dev_filelist = REPO / "data" / "librispeech_dev.txt"
    if not dev_filelist.exists():
        print(f"Error: Dev filelist not found at {dev_filelist}")
        return

    chosen = pick_samples(dev_filelist, args.n_samples, args.seed)
    print(f"Evaluating {len(chosen)} samples (DFR ratio={args.down_sample_ratio}, max_comp={args.max_compression})")

    results = []
    for fid, relpath in tqdm(chosen):
        wavpath = REPO / "data" / relpath
        if not wavpath.exists():
            print(f"Warning: {wavpath} not found, skipping")
            continue

        wav, _ = librosa.load(str(wavpath), sr=sr)
        # Pad to multiple of 200 (encoder hop length)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        pad = (200 - wav_t.shape[1] % 200) % 200
        if pad:
            wav_t = F.pad(wav_t, (0, pad))
        wav_t = wav_t.to(args.device)

        with torch.no_grad():
            # Encode: [1, T] -> [1, 1, T] -> [1, C, T_enc]
            vq_emb = encoder(wav_t.unsqueeze(1))

            # Full reconstruction
            vq_post, _, _ = decoder(vq_emb, vq=True)
            recon_full = decoder(vq_post, vq=False).squeeze(0).squeeze(0).cpu().numpy()

            # COOL: DFR needs [T_enc, C] (time-first)
            tokens = vq_emb.squeeze(0).T.cpu().float().numpy()  # [T_enc, C]
            encoded = dfr.optimal_down_sample(tokens)
            tokens_comp = dfr.up_sample_encoded(encoded)        # [T', C]
            comp_ratio = len(tokens) / len(encoded.encoded_data)
            encoding_lengths = encoded.encoding_lengths

            # Decode compressed: [T', C] -> [1, C, T'] for decoder
            t_comp = torch.from_numpy(tokens_comp.T).float().unsqueeze(0).to(args.device)
            vq_post_c, _, _ = decoder(t_comp, vq=True)
            recon_cool = decoder(vq_post_c, vq=False).squeeze(0).squeeze(0).cpu().numpy()

        recon_full = recon_full[:len(wav)]
        recon_cool = recon_cool[:len(wav)]

        metrics_full = calculate_metrics(wav, recon_full, sr)
        metrics_cool = calculate_metrics(wav, recon_cool, sr)

        sf.write(args.out_dir / f"{fid}_orig.wav", wav, sr)
        sf.write(args.out_dir / f"{fid}_full_recon.wav", recon_full, sr)
        sf.write(args.out_dir / f"{fid}_cool_recon.wav", recon_cool, sr)

        results.append({
            "fid": fid,
            "compression_ratio": comp_ratio,
            "encoding_lengths": encoding_lengths,
            "metrics_full": metrics_full,
            "metrics_cool": metrics_cool,
            "duration_s": metrics_full["duration_s"],
        })
        print(f"  {fid}: ratio={comp_ratio:.2f}, full_SNR={metrics_full['snr_db']:.1f}dB, "
              f"cool_SNR={metrics_cool['snr_db']:.1f}dB")

    summary_path = args.out_dir / "cool_evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("COOL Evaluation Summary\n")
        f.write(f"Checkpoint: {args.ckpt.name}\n")
        f.write(f"DFR ratio: {args.down_sample_ratio}\n")
        f.write(f"Max compression: {args.max_compression}\n")
        f.write(f"Samples: {len(results)}\n\n")
        avg_comp = np.mean([r["compression_ratio"] for r in results])
        avg_snr_full = np.mean([r["metrics_full"]["snr_db"] for r in results])
        avg_snr_cool = np.mean([r["metrics_cool"]["snr_db"] for r in results])
        f.write(f"Average compression ratio: {avg_comp:.2f}\n")
        f.write(f"Average SNR (full): {avg_snr_full:.2f} dB\n")
        f.write(f"Average SNR (COOL): {avg_snr_cool:.2f} dB\n")
        f.write(f"SNR degradation: {avg_snr_full - avg_snr_cool:.2f} dB\n\n")
        f.write("Per-sample results:\n")
        for r in results:
            f.write(f"{r['fid']}: ratio={r['compression_ratio']:.2f}, "
                    f"full_SNR={r['metrics_full']['snr_db']:.1f}dB, "
                    f"cool_SNR={r['metrics_cool']['snr_db']:.1f}dB\n")

    print(f"\nDone. Results saved to {args.out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()