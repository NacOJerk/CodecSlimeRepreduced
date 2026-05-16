"""Integration test: CodecLightningModule must construct cleanly under both
the VQ and FSQ model configs, exercising the full chain
(config -> lightning_module.construct_model -> CodecEncoder/Decoder).

Earlier a unit test on CodecDecoder alone passed while the FSQ run failed
because lightning_module read VQ-specific keys unconditionally; this test
covers that integration.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "BigCodec"))

import pytest
from omegaconf import OmegaConf
from lightning_module import CodecLightningModule


def _build_cfg(model_yaml: str) -> OmegaConf:
    cfg = OmegaConf.create({})
    cfg.model = OmegaConf.load(REPO_ROOT / "backbones" / "configs" / "model" / f"{model_yaml}.yaml")
    cfg.train = OmegaConf.load(REPO_ROOT / "backbones" / "configs" / "train" / "codecslime_300k.yaml")
    cfg.dataset = OmegaConf.load(REPO_ROOT / "backbones" / "configs" / "dataset" / "librispeech.yaml")
    cfg.preprocess = OmegaConf.create({"audio": {"sr": 16000}})
    return cfg


@pytest.mark.parametrize("model_yaml,expected_quantizer,expected_codebook", [
    ("vq8k", "ResidualVQ", 8192),
    ("fsq18k", "FSQQuantizer", 18225),
])
def test_lightning_module_constructs(model_yaml, expected_quantizer, expected_codebook, monkeypatch):
    import hydra.utils
    monkeypatch.setattr(hydra.utils, "get_original_cwd", lambda: str(REPO_ROOT))
    cfg = _build_cfg(model_yaml)
    lm = CodecLightningModule(cfg)
    decoder = lm.model["generator"]
    assert type(decoder.quantizer).__name__ == expected_quantizer
    if expected_quantizer == "FSQQuantizer":
        assert decoder.quantizer.codebook_size == expected_codebook
