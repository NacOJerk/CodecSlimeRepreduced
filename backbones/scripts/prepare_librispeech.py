"""Download LibriSpeech subsets to a target dir and extract."""
import argparse
import subprocess
from pathlib import Path

URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="parent dir; LibriSpeech/ will appear inside")
    ap.add_argument("--subsets", nargs="+", default=list(URLS.keys()))
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.subsets:
        tarball = args.root / f"{name}.tar.gz"
        marker = args.root / "LibriSpeech" / name
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
