"""Encode + decode a handful of LibriSpeech utterances using a trained
backbone checkpoint, and save originals + reconstructions side-by-side.

Usage:
    python reconstruct_samples.py \\
        --ckpt path/to/last.ckpt \\
        --config codecslime_vq8k \\
        --out-dir backbones/results/recon-vq8k

Drops the originals as `<id>_orig.wav` and reconstructions as
`<id>_recon.wav` next to each other so listening A/B is easy.
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

REPO = Path("/home/morg/students/dortirosh/audio_ml_tau_final")
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
                "train.logger.name=recon",
                "train.logger.id=recon",
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


def reconstruct(lm, wav: np.ndarray, sr: int) -> np.ndarray:
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).cuda()
    pad = 200 - (wav_t.shape[1] % 200)
    if pad and pad != 200:
        wav_t = F.pad(wav_t, (0, pad))
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        vq_emb = lm.model["CodecEnc"](wav_t.unsqueeze(1))
        vq_post, _, _ = lm.model["generator"](vq_emb, vq=True)
        recon = lm.model["generator"](vq_post, vq=False)
    return recon.squeeze(0).squeeze(0).float().cpu().numpy()[: len(wav)]


def pick_samples(filelist: Path, n: int, seed: int) -> list:
    lines = [ln.strip().split("|") for ln in filelist.read_text().splitlines() if ln.strip()]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(lines), size=min(n, len(lines)), replace=False)
    return [lines[i] for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, choices=["codecslime_vq8k", "codecslime_fsq18k"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-train", type=int, default=3, help="samples from train-clean-100/360/500")
    ap.add_argument("--n-dev", type=int, default=3, help="samples from dev-clean")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.config)
    lm = load_module(args.ckpt, cfg)

    sr = cfg.preprocess.audio.sr
    root = REPO / "datasets" / "LibriSpeech"

    splits = [
        ("train", REPO / "backbones" / "data" / "librispeech_train.txt", args.n_train),
        ("dev", REPO / "backbones" / "data" / "librispeech_test.txt", args.n_dev),
    ]
    for split_name, filelist, n in splits:
        if n <= 0:
            continue
        if not filelist.exists():
            print(f"[skip {split_name}] no filelist at {filelist}")
            continue
        chosen = pick_samples(filelist, n, args.seed + (0 if split_name == "train" else 1))
        for fid, relpath in chosen:
            wavpath = root / relpath
            wav, _ = librosa.load(wavpath, sr=sr)
            recon = reconstruct(lm, wav, sr)
            orig_path = args.out_dir / f"{split_name}_{fid}_orig.wav"
            recon_path = args.out_dir / f"{split_name}_{fid}_recon.wav"
            sf.write(orig_path, wav, sr)
            sf.write(recon_path, recon, sr)
            print(f"  {split_name}/{fid}: {orig_path.name} ({len(wav)/sr:.1f}s)")

    print(f"\nDone. Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
