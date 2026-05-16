"""FSQ with levels [3,3,3,3,3,3,5,5] should yield exactly 18225 codes."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))

from vq.fsq_quantizer import FSQQuantizer


def test_codebook_size_18225():
    q = FSQQuantizer(input_dim=1024, fsq_dim=8, levels=[3, 3, 3, 3, 3, 3, 5, 5])
    assert q.codebook_size == 18225
