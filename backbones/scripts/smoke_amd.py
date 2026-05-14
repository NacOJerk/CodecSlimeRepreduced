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
from hydra import compose, initialize_config_dir

REPO_ROOT = Path("/home/morg/students/dortirosh/audio_ml_tau_final")
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
    cfg_dir = str(REPO_ROOT / "backbones" / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=f"codecslime_melt_{backbone}")
    return cfg


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

    B, T = 2, 16000
    wav = torch.randn(B, T, device="cuda")
    batch = {"wav": wav}

    t0 = time.perf_counter()
    out = model(batch)
    print(f"  forward done in {time.perf_counter() - t0:.3f}s; gen_wav shape={tuple(out['gen_wav'].shape)}")

    loss = (out["gen_wav"] - wav.unsqueeze(1)).pow(2).mean() + out["vq_loss"]
    t0 = time.perf_counter()
    loss.backward()
    print(f"  backward done in {time.perf_counter() - t0:.3f}s; loss={loss.item():.4f}")

    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"  params with grads: {n_grad}/{n_total}")
    if n_grad == 0:
        print("ERROR: no gradients flowed")
        sys.exit(2)


if __name__ == "__main__":
    report_gpu()
    backbones = sys.argv[1:] or ["vq8k", "fsq18k"]
    for bb in backbones:
        forward_backward(bb)
    print("\nALL SMOKE OK")
