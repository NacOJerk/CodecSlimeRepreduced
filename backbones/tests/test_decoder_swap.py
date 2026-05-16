"""CodecDecoder must accept quantizer_type='fsq' and produce expected shapes."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))

import torch
from vq.codec_decoder import CodecDecoder


def test_vq_decoder_default():
    dec = CodecDecoder()  # quantizer_type defaults to "vq"
    assert hasattr(dec, "quantizer")


def test_fsq_decoder_swap_returns_post_q_features():
    """vq=True returns post-quantizer features (no upsample); the lightning
    module then calls forward(features, vq=False) to upsample them."""
    dec = CodecDecoder(
        quantizer_type="fsq",
        fsq_levels=[3, 3, 3, 3, 3, 3, 5, 5],
        fsq_dim=8,
    )
    x = torch.randn(1, 1024, 80)  # (B, C, T) at 80 Hz
    post_q, indices, commit = dec(x, vq=True)
    assert post_q.shape == x.shape, f"expected {x.shape}, got {post_q.shape}"
    assert commit.item() == 0.0


def test_fsq_decoder_full_pipeline():
    """Two-call pattern: vq=True quantizes, vq=False upsamples 80 Hz -> 16 kHz."""
    dec = CodecDecoder(
        quantizer_type="fsq",
        fsq_levels=[3, 3, 3, 3, 3, 3, 5, 5],
        fsq_dim=8,
    )
    x = torch.randn(1, 1024, 80)
    post_q, _, _ = dec(x, vq=True)
    audio = dec(post_q, vq=False)
    # decoder up_ratios product = 5*5*2*2*2 = 200, so 80 frames -> 16000 samples
    assert audio.shape[-1] == 80 * 200
