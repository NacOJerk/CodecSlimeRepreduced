"""Download LibriTTS subsets to a target dir and extract."""
import argparse
import subprocess
from pathlib import Path

URLS = {
    "dev-clean": "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/60/test-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/60/dev-other.tar.gz",
    "test-other": "https://www.openslr.org/resources/60/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/60/train-other-500.tar.gz",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="parent dir; LibriTTS/ will appear inside")
    ap.add_argument("--subsets", nargs="+", default=["test-clean"])
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.subsets:
        tarball = args.root / f"libritts-{name}.tar.gz"
        marker = args.root / "LibriTTS" / name
        if marker.exists():
            print(f"[skip] {name} already extracted")
            continue
        if not tarball.exists():
            print(f"[get] {name}")
            subprocess.check_call(["wget", "-c", URLS[name], "-O", str(tarball)])
        print(f"[extract] {name}")
        subprocess.check_call(["tar", "-xzf", str(tarball), "-C", str(args.root)])
        tarball.unlink()


if __name__ == "__main__":
    main()
