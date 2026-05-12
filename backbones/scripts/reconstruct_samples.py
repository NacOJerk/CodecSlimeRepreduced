"""Encode + decode LibriSpeech utterances using a trained backbone checkpoint.

Usage:
    python reconstruct_samples.py \\
        --ckpt models/fsq18k-300k-inference.ckpt \\
        --out-dir backbones/results/vanilla-fsq18k \\
        --n-samples 10
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "BigCodec"))

from omegaconf import OmegaConf
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder


def load_codec(ckpt_path: Path, device: str):
    is_fsq = "fsq" in ckpt_path.name
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
    return encoder, decoder


def reconstruct(encoder, decoder, wav: np.ndarray, device: str) -> np.ndarray:
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
    pad = (200 - wav_t.shape[1] % 200) % 200
    if pad:
        wav_t = F.pad(wav_t, (0, pad))
    wav_t = wav_t.to(device)
    with torch.no_grad():
        vq_emb = encoder(wav_t.unsqueeze(1))         # [1, 1, T] -> [1, C, T_enc]
        vq_post, _, _ = decoder(vq_emb, vq=True)
        recon = decoder(vq_post, vq=False)            # [1, 1, T_out]
    return recon.squeeze(0).squeeze(0).cpu().float().numpy()


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
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder = load_codec(args.ckpt, args.device)
    sr = 16000

    dev_filelist = REPO / "data" / "librispeech_dev.txt"
    if not dev_filelist.exists():
        print(f"Error: Dev filelist not found at {dev_filelist}")
        return

    chosen = pick_samples(dev_filelist, args.n_samples, args.seed)
    print(f"Reconstructing {len(chosen)} samples")

    results = []
    for fid, relpath in tqdm(chosen):
        wavpath = REPO / "data" / relpath
        if not wavpath.exists():
            print(f"Warning: {wavpath} not found, skipping")
            continue

        wav, _ = librosa.load(str(wavpath), sr=sr)
        recon = reconstruct(encoder, decoder, wav, args.device)
        recon = recon[:len(wav)]

        sf.write(args.out_dir / f"{fid}_orig.wav", wav, sr)
        sf.write(args.out_dir / f"{fid}_recon.wav", recon, sr)

        mse = np.mean((wav - recon) ** 2)
        snr = 10 * np.log10(np.mean(wav ** 2) / mse) if mse > 0 else float("inf")
        results.append({"fid": fid, "snr_db": snr})
        print(f"  {fid}: SNR={snr:.1f}dB")

    summary_path = args.out_dir / "reconstruction_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Reconstruction Summary\n")
        f.write(f"Checkpoint: {args.ckpt.name}\n")
        f.write(f"Samples: {len(results)}\n\n")
        avg_snr = np.mean([r["snr_db"] for r in results])
        f.write(f"Average SNR: {avg_snr:.2f} dB\n\n")
        f.write("Per-sample results:\n")
        for r in results:
            f.write(f"{r['fid']}: SNR={r['snr_db']:.1f}dB\n")

    print(f"\nDone. Results saved to {args.out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
