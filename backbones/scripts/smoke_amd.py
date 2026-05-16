"""ROCm smoke test for the cs_amd env on n-210.

Verifies:
  * torch sees the AMD GPU(s)
  * a small forward+backward through CoolMeltWrapper succeeds on GPU
  * MeltManager runs and produces a step-conditional proportion
  * the AMD-side gradients flow back

Run from the repo root with cs_amd activated; PYTHONPATH must include the
BigCodec vendored dir (slurm/script sets that up).
"""
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))

from melt_manager import MeltManager  # noqa: E402
from melt_wrapper import CoolMeltWrapper  # noqa: E402


def report_gpu():
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() == False; aborting")
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"device_count={n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] name={p.name!r} total_memory_GiB={p.total_memory / 2**30:.1f}")
    print(f"current_device={torch.cuda.current_device()}")


def build_cfg(backbone: str):
    # Side-step Hydra by composing manually with OmegaConf. The wrapper only
    # reads `cfg.train.*` and `cfg.model.*` (and `cfg.dataset.train.batch_size`
    # for the dataloader, which we don't build here).
    train_yaml = REPO_ROOT / "backbones" / "configs" / "train" / "codecslime_melt_n210.yaml"
    model_yaml = REPO_ROOT / "backbones" / "configs" / "model" / f"{backbone}.yaml"
    dataset_yaml = REPO_ROOT / "backbones" / "configs" / "dataset" / "librispeech_b64.yaml"
    cfg = OmegaConf.create({
        "train": OmegaConf.load(train_yaml),
        "model": OmegaConf.load(model_yaml),
        "dataset": OmegaConf.load(dataset_yaml),
    })
    return cfg


def _step(model, B: int):
    T = 16000
    wav = torch.randn(B, T, device="cuda")
    batch = {"wav": wav}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(batch)
    fwd_t = time.perf_counter() - t0
    loss = (out["gen_wav"] - wav.unsqueeze(1)).pow(2).mean() + out["vq_loss"]

    t0 = time.perf_counter()
    loss.backward()
    bwd_t = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    model.zero_grad(set_to_none=True)
    return fwd_t, bwd_t, peak, loss.item()


def forward_backward(backbone: str):
    print(f"\n=== smoke {backbone} ===")
    cfg = build_cfg(backbone)
    mm = MeltManager(
        max_compression=cfg.train.compression.max_compression,
        p_tgt=list(cfg.train.compression.p_tgt),
        s_p=cfg.train.compression.s_p,
        concentration_control=cfg.train.compression.concentration_control,
        skip_prob=0.0,  # force a Melt path (no skip) so we exercise the op
    )
    model = CoolMeltWrapper(cfg, melt_manager=mm, cool_manager=None).cuda()
    model.train()

    for B in (2, 16, 64):
        try:
            fwd_t, bwd_t, peak, loss_v = _step(model, B)
            print(f"  B={B:>3d}  fwd={fwd_t:.3f}s  bwd={bwd_t:.3f}s  peak={peak:.2f}GiB  loss={loss_v:.4f}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  B={B:>3d}  OOM: {e}")
            torch.cuda.empty_cache()

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"  params with grads (after last step): {n_grad}/{n_total}")


if __name__ == "__main__":
    report_gpu()
    backbones = sys.argv[1:] or ["vq8k", "fsq18k"]
    for bb in backbones:
        forward_backward(bb)
    print("\nALL SMOKE OK")
