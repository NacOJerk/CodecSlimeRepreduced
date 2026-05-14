"""Minimal ROCm smoke. Prints status with explicit flushing so we see progress."""
from pathlib import Path
import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


REPO_ROOT = Path("/home/morg/students/dortirosh/audio_ml_tau_final")


def main():
    log("[smoke_min] start")
    log("[smoke_min] importing torch ...")
    t0 = time.perf_counter()
    import torch
    log(f"[smoke_min] torch={torch.__version__} hip={torch.version.hip} (import took {time.perf_counter()-t0:.1f}s)")

    log("[smoke_min] checking cuda.is_available()")
    t0 = time.perf_counter()
    available = torch.cuda.is_available()
    log(f"[smoke_min] is_available={available} (took {time.perf_counter()-t0:.1f}s)")
    if not available:
        log("[smoke_min] FAIL: no AMD GPU visible to torch")
        sys.exit(1)

    n = torch.cuda.device_count()
    log(f"[smoke_min] device_count={n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        log(f"[smoke_min]   [{i}] name={p.name!r} mem={p.total_memory/2**30:.1f}GiB")

    log("[smoke_min] tiny tensor forward + backward")
    t0 = time.perf_counter()
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    y = (x @ x.T).sum()
    y.backward()
    torch.cuda.synchronize()
    log(f"[smoke_min] tiny test done in {time.perf_counter()-t0:.2f}s")

    log("[smoke_min] importing project modules ...")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))
    t0 = time.perf_counter()
    from melt_wrapper import CoolMeltWrapper  # noqa: F401
    from melt_manager import MeltManager  # noqa: F401
    log(f"[smoke_min] project imports done in {time.perf_counter()-t0:.1f}s")

    log("[smoke_min] composing config (manual OmegaConf)")
    from omegaconf import OmegaConf
    train = OmegaConf.load(REPO_ROOT / "backbones" / "configs" / "train" / "codecslime_melt_n210.yaml")
    model_vq = OmegaConf.load(REPO_ROOT / "backbones" / "configs" / "model" / "vq8k.yaml")
    cfg = OmegaConf.create({"train": train, "model": model_vq})
    log("[smoke_min] cfg ready")

    mm = MeltManager(
        max_compression=cfg.train.compression.max_compression,
        p_tgt=list(cfg.train.compression.p_tgt),
        s_p=cfg.train.compression.s_p,
        concentration_control=cfg.train.compression.concentration_control,
        skip_prob=0.0,
    )

    log("[smoke_min] building CoolMeltWrapper (vq8k)")
    t0 = time.perf_counter()
    model = CoolMeltWrapper(cfg, melt_manager=mm, cool_manager=None).cuda()
    model.train()
    log(f"[smoke_min] model built in {time.perf_counter()-t0:.1f}s")

    for B in (2, 16, 64):
        try:
            T = 16000
            wav = torch.randn(B, T, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = model({"wav": wav})
            fwd = time.perf_counter() - t0
            loss = (out["gen_wav"] - wav.unsqueeze(1)).pow(2).mean() + out["vq_loss"]
            t0 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            bwd = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"[smoke_min] B={B:>3d}  fwd={fwd:.3f}s  bwd={bwd:.3f}s  peak={peak:.2f}GiB  loss={loss.item():.4f}")
            model.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as e:
            log(f"[smoke_min] B={B:>3d}  OOM: {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"[smoke_min] B={B:>3d}  EXC: {type(e).__name__}: {e}")
            raise

    log("[smoke_min] DONE")


if __name__ == "__main__":
    main()
