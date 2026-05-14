"""Build the UniCATS testset B manifest from the official upstream file.

UniCATS testset B is 500 utterances from 37 unseen speakers of LibriTTS
test-clean (Du et al., AAAI 2024). The canonical utt-to-prompt list lives at
`https://cpdu.github.io/unicats/resources/testsetB_utt2prompt`. This script
downloads it and emits `unicats_b.txt` in the repo's `fid|relpath` manifest
convention, with `relpath` resolving under `datasets/LibriTTS/`.
"""
import argparse
import subprocess
from pathlib import Path

URL = "https://cpdu.github.io/unicats/resources/testsetB_utt2prompt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory to write unicats_b.txt + unicats_b_utt2prompt.txt")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    upstream_path = args.out_dir / "unicats_b_utt2prompt.txt"
    manifest_path = args.out_dir / "unicats_b.txt"

    print(f"[get] {URL} -> {upstream_path}")
    subprocess.check_call(["curl", "-fsSL", "-o", str(upstream_path), URL])

    rows = []
    speakers = set()
    for ln in upstream_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        utt = ln.split()[0]
        spk, chp, *_ = utt.split("_")
        speakers.add(spk)
        rows.append(f"{utt}|test-clean/{spk}/{chp}/{utt}.wav")

    if len(rows) != 500:
        raise RuntimeError(f"expected 500 utterances, got {len(rows)}")
    if len(speakers) != 37:
        raise RuntimeError(f"expected 37 speakers, got {len(speakers)}")

    manifest_path.write_text("\n".join(rows) + "\n")
    print(f"[ok] wrote {manifest_path} ({len(rows)} utts, {len(speakers)} speakers)")


if __name__ == "__main__":
    main()
