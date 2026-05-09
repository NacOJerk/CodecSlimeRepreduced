import sys
import torch
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

from vq.fsq_quantizer import FSQQuantizer


def test_forward_shape_and_indices_range():
    torch.manual_seed(0)
    q = FSQQuantizer(input_dim=1024, fsq_dim=8, levels=[3, 3, 3, 3, 3, 3, 5, 5])
    x = torch.randn(2, 50, 1024)
    out, indices, commit_loss = q(x)
    assert out.shape == (2, 50, 1024)
    assert indices.shape[0] == 2 and indices.shape[1] == 50
    assert indices.min().item() >= 0
    assert indices.max().item() < 18225
    assert commit_loss.item() == 0.0
