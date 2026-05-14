"""Build the final results bundle: a clean CSV + markdown table + plots.

Reads each `backbones/results/eval-<variant>/metrics_summary.json`, parses
the variant name into (stage, backbone, mode), and writes everything to
`backbones/results/final/`.

Plots (all PNG, 150 DPI):
    wer_by_variant.png     grouped bar (variants, by backbone)
    pesq_by_variant.png    same for PESQ
    utmos_by_variant.png   same for UTMOS
    metrics_grid.png       2x3 grid of all five metrics + bitrate
    quality_vs_bitrate.png scatter of WER and PESQ against bitrate
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Stage / backbone / mode parsing rules. Order matters: longest-prefix match
# wins so meltcool catches before melt, etc.
STAGE_RULES = [
    ("coolmelt-",                    "Melt+Cool"),
    ("cool-vq8k-from-backbone-",     "Cool only"),
    ("cool-fsq18k-from-backbone-",   "Cool only"),
    ("melt-vq8k-n210-paperd-",       "Melt (paperd)"),
    ("melt-",                        "Melt"),
    ("backbone-",                    "Backbone"),
]
MODE_LABELS = {"ffr": "FFR 80Hz", "fixed": "Fixed 40Hz", "dfr": "ScheDFR 40Hz"}
STAGE_ORDER = ["Backbone", "Melt", "Melt (paperd)", "Cool only", "Melt+Cool"]
MODE_ORDER = ["FFR 80Hz", "Fixed 40Hz", "ScheDFR 40Hz"]
BACKBONE_ORDER = ["VQ-8k", "FSQ-18k"]
COLORS_STAGE = {
    "Backbone":      "#7f7f7f",
    "Melt":          "#1f77b4",
    "Melt (paperd)": "#9467bd",
    "Cool only":     "#2ca02c",
    "Melt+Cool":     "#d62728",
}


def parse_variant(variant: str):
    for prefix, stage in STAGE_RULES:
        if variant.startswith(prefix):
            break
    else:
        stage = "?"
    backbone = "VQ-8k" if "vq8k" in variant else ("FSQ-18k" if "fsq18k" in variant else "?")
    if "ffr80" in variant:
        mode = MODE_LABELS["ffr"]
    elif "ffr40" in variant:
        mode = MODE_LABELS["fixed"]
    elif "dfr40" in variant:
        mode = MODE_LABELS["dfr"]
    else:
        mode = "?"
    return stage, backbone, mode


def load_rows(results_dir: Path):
    rows = []
    for d in sorted(results_dir.glob("eval-*")):
        s = d / "metrics_summary.json"
        if not s.exists():
            continue
        meta = json.loads(s.read_text())
        a = meta["averages"]
        variant = d.name.replace("eval-", "", 1)
        stage, backbone, mode = parse_variant(variant)
        rows.append({
            "Stage": stage,
            "Backbone": backbone,
            "Mode": mode,
            "Bitrate (bps)": round(a["bitrate_bps"]),
            "WER": a["wer"],
            "STOI": a["stoi"],
            "PESQ": a["pesq"],
            "SECS": a["secs"],
            "UTMOS": a["utmos"],
            "_variant": variant,
        })
    df = pd.DataFrame(rows)
    df["Stage"] = pd.Categorical(df["Stage"], STAGE_ORDER, ordered=True)
    df["Backbone"] = pd.Categorical(df["Backbone"], BACKBONE_ORDER, ordered=True)
    df["Mode"] = pd.Categorical(df["Mode"], MODE_ORDER, ordered=True)
    return df.sort_values(["Backbone", "Stage", "Mode"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    pretty = df.drop(columns=["_variant"]).copy()
    for c in ["WER", "STOI", "PESQ", "SECS", "UTMOS"]:
        pretty[c] = pretty[c].map(lambda v: f"{v:.4f}")
    pretty.to_csv(out, index=False)


def write_markdown(df: pd.DataFrame, out: Path):
    rename = {"Bitrate (bps)": "Bitrate", "WER": "WER ↓",
              "STOI": "STOI ↑", "PESQ": "PESQ ↑",
              "SECS": "SECS ↑", "UTMOS": "UTMOS ↑"}
    pretty = df.drop(columns=["_variant"]).rename(columns=rename).copy()
    for c in ["WER ↓", "STOI ↑", "PESQ ↑", "SECS ↑", "UTMOS ↑"]:
        pretty[c] = pretty[c].map(lambda v: f"{v:.4f}")
    out.write_text(pretty.to_markdown(index=False) + "\n")


def _row_label(r):
    return f"{r['Backbone']} | {r['Stage']} | {r['Mode']}"


def _bar(df: pd.DataFrame, metric: str, lower_better: bool, out: Path, title: str):
    sub = df.sort_values(["Backbone", "Stage", "Mode"])
    labels = [_row_label(r) for _, r in sub.iterrows()]
    values = sub[metric].to_numpy()
    colors = [COLORS_STAGE[s] for s in sub["Stage"]]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(sub) + 2)))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel(metric + (" (lower is better)" if lower_better else " (higher is better)"))
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2,
                f" {v:.3f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS_STAGE[s])
               for s in STAGE_ORDER if s in set(sub["Stage"])]
    ax.legend(handles, [s for s in STAGE_ORDER if s in set(sub["Stage"])],
              loc="lower right", title="Stage", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _grid(df: pd.DataFrame, out: Path):
    metrics = [("Bitrate (bps)", False), ("WER", True), ("STOI", False),
               ("PESQ", False), ("SECS", False), ("UTMOS", False)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sub = df.sort_values(["Backbone", "Stage", "Mode"])
    labels = [_row_label(r) for _, r in sub.iterrows()]
    colors = [COLORS_STAGE[s] for s in sub["Stage"]]
    for ax, (metric, lower) in zip(axes.flat, metrics):
        ax.barh(labels, sub[metric].to_numpy(), color=colors, edgecolor="black",
                linewidth=0.4)
        ax.invert_yaxis()
        ax.set_xlabel(metric + (" (lower is better)" if lower else ""))
        ax.set_title(metric)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("CodecSlime evaluation matrix on UniCATS-B (500 utts)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _quality_vs_bitrate(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (metric, lower) in zip(axes, [("WER", True), ("PESQ", False)]):
        for stage in STAGE_ORDER:
            for backbone in BACKBONE_ORDER:
                sub = df[(df["Stage"] == stage) & (df["Backbone"] == backbone)]
                if sub.empty:
                    continue
                marker = "o" if backbone == "VQ-8k" else "s"
                ax.scatter(sub["Bitrate (bps)"], sub[metric],
                           s=110, marker=marker, color=COLORS_STAGE[stage],
                           edgecolor="black", linewidth=0.6,
                           label=f"{stage} | {backbone}")
                for _, r in sub.iterrows():
                    ax.annotate(r["Mode"].split()[0],
                                (r["Bitrate (bps)"], r[metric]),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=7, alpha=0.8)
        ax.set_xlabel("Bitrate (bps)")
        ax.set_ylabel(metric + (" (lower is better)" if lower else " (higher is better)"))
        ax.set_title(f"{metric} vs bitrate")
        ax.grid(linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(*zip(*uniq), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle("Quality vs bitrate trade-off", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("backbones/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("backbones/results/final"))
    args = ap.parse_args()

    df = load_rows(args.results_dir)
    if df.empty:
        raise SystemExit(f"no eval-* metrics found under {args.results_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(df, args.out_dir / "all_metrics.csv")
    write_markdown(df, args.out_dir / "all_metrics.md")

    _bar(df, "WER", True, args.out_dir / "wer_by_variant.png", "WER per cell")
    _bar(df, "PESQ", False, args.out_dir / "pesq_by_variant.png", "PESQ per cell")
    _bar(df, "UTMOS", False, args.out_dir / "utmos_by_variant.png", "UTMOS per cell")
    _grid(df, args.out_dir / "metrics_grid.png")
    _quality_vs_bitrate(df, args.out_dir / "quality_vs_bitrate.png")

    print(f"[ok] wrote {len(df)} rows to {args.out_dir}/")
    for p in sorted(args.out_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
