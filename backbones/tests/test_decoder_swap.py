"""CodecDecoder must accept quantizer_type='fsq' and produce expected shapes."""
import sys
import torch
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

from vq.codec_decoder import CodecDecoder


def test_vq_decoder_default():
    dec = CodecDecoder()  # quantizer_type defaults to "vq"
    assert hasattr(dec, "quantizer")


def test_fsq_decoder_swap():
    dec = CodecDecoder(
        quantizer_type="fsq",
        fsq_levels=[3, 3, 3, 3, 3, 3, 5, 5],
        fsq_dim=8,
    )
    # CodecDecoder typically expects encoder output as (B, C, T)
    x = torch.randn(1, 1024, 80)
    out, indices, commit = dec(x, vq=True)
    # decoder upsample product is 5*5*2*2*2 = 200, so 80 frames -> 16000 samples
    assert out.shape[-1] == 80 * 200
    assert commit.item() == 0.0
