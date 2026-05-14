"""Merge per-cell metrics_summary.json files into one detailed CSV.

Walks `<results-dir>/eval-*/metrics_summary.json` and produces a flat CSV with
one row per evaluation cell, sorted by variant. Safe to run before all jobs
have finished: missing or in-progress cells are simply skipped.
"""
import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "variant", "model_name", "mode", "ckpt", "n_utt",
    "bitrate_bps", "comp_ratio", "duration_bits_per_frame",
    "wer", "stoi", "pesq", "secs", "utmos",
    "rs", "u", "manifest_sha1",
]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("backbones/results"),
                    help="dir containing eval-<variant>/ subdirs")
    ap.add_argument("--out", type=Path, default=Path("backbones/results/all_metrics.csv"))
    ap.add_argument("--pattern", default="eval-*",
                    help="subdir glob; default eval-* skips smoke-eval-*")
    args = ap.parse_args()

    rows = []
    for d in sorted(args.results_dir.glob(args.pattern)):
        if not d.is_dir():
            continue
        summary = d / "metrics_summary.json"
        if not summary.exists():
            print(f"[skip] {d.name}: no metrics_summary.json yet")
            continue
        try:
            s = json.loads(summary.read_text())
        except Exception as exc:
            print(f"[skip] {d.name}: cannot parse JSON ({exc})")
            continue

        variant = d.name.replace("eval-", "", 1)
        a = s.get("args", {})
        avg = s.get("averages", {})

        rows.append({
            "variant": variant,
            "model_name": s.get("model_name"),
            "mode": a.get("mode"),
            "ckpt": Path(a.get("ckpt", "")).parent.name or a.get("ckpt", ""),
            "n_utt": avg.get("n_utt"),
            "bitrate_bps": avg.get("bitrate_bps"),
            "comp_ratio": avg.get("comp_ratio"),
            "duration_bits_per_frame": avg.get("duration_bits_per_frame"),
            "wer": avg.get("wer"),
            "stoi": avg.get("stoi"),
            "pesq": avg.get("pesq"),
            "secs": avg.get("secs"),
            "utmos": avg.get("utmos"),
            "rs": a.get("rs"),
            "u": a.get("u"),
            "manifest_sha1": (s.get("manifest_sha1") or "")[:10],
        })

    rows.sort(key=lambda r: r["variant"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})

    print(f"[ok] wrote {args.out} ({len(rows)} rows)")
    if rows:
        # Print a compact table to stdout
        widths = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0)) for c in COLUMNS}
        print()
        print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
        for r in rows:
            print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in COLUMNS))


if __name__ == "__main__":
    main()
