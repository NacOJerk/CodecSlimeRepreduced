"""Load a BigCodec encoder + decoder from one of our Lightning checkpoints.

Shared by reconstruction / evaluation scripts. The encoder and decoder are
built from `backbones/configs/model/{vq8k,fsq18k}.yaml`; the model variant is
inferred from the checkpoint name (`"fsq" in ckpt_path.name` -> fsq18k, else
vq8k). Weights are loaded from the Lightning state_dict by stripping the
`model.CodecEnc.` and `model.generator.` prefixes used by both backbone
(`CodecLightningModule`) and Melt (`CoolMeltWrapper`) checkpoints.
"""
import sys
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "external" / "BigCodec"))

from vq.codec_encoder import CodecEncoder  # noqa: E402
from vq.codec_decoder import CodecDecoder  # noqa: E402


def infer_model_name(ckpt_path: Path) -> str:
    return "fsq18k" if "fsq" in ckpt_path.name.lower() else "vq8k"


def load_codec(ckpt_path: Path, device: str) -> Tuple[CodecEncoder, CodecDecoder, str]:
    model_name = infer_model_name(ckpt_path)
    cfg = OmegaConf.load(REPO / "backbones" / "configs" / "model" / f"{model_name}.yaml")

    enccfg = cfg.codec_encoder
    encoder = CodecEncoder(
        ngf=enccfg.ngf,
        use_rnn=enccfg.use_rnn,
        rnn_bidirectional=enccfg.rnn_bidirectional,
        rnn_num_layers=enccfg.rnn_num_layers,
        up_ratios=list(enccfg.up_ratios),
        dilations=list(enccfg.dilations),
        out_channels=enccfg.out_channels,
    )

    deccfg = cfg.codec_decoder
    quantizer_type = deccfg.get("quantizer_type", "vq")
    dec_kwargs = dict(
        in_channels=deccfg.in_channels,
        upsample_initial_channel=deccfg.upsample_initial_channel,
        ngf=deccfg.ngf,
        use_rnn=deccfg.use_rnn,
        rnn_bidirectional=deccfg.rnn_bidirectional,
        rnn_num_layers=deccfg.rnn_num_layers,
        up_ratios=list(deccfg.up_ratios),
        dilations=list(deccfg.dilations),
        vq_dim=deccfg.vq_dim,
        quantizer_type=quantizer_type,
    )
    if quantizer_type == "fsq":
        dec_kwargs.update(fsq_levels=list(deccfg.fsq_levels), fsq_dim=deccfg.fsq_dim)
    else:
        dec_kwargs.update(
            vq_num_quantizers=deccfg.vq_num_quantizers,
            vq_commit_weight=deccfg.vq_commit_weight,
            vq_full_commit_loss=deccfg.vq_full_commit_loss,
            codebook_size=deccfg.codebook_size,
            codebook_dim=deccfg.codebook_dim,
        )
    decoder = CodecDecoder(**dec_kwargs)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[len("model.CodecEnc."):]: v for k, v in sd.items() if k.startswith("model.CodecEnc.")}
    dec_sd = {k[len("model.generator."):]: v for k, v in sd.items() if k.startswith("model.generator.")}
    if not enc_sd or not dec_sd:
        raise RuntimeError(
            f"Unexpected state_dict layout in {ckpt_path}: "
            f"enc_keys={len(enc_sd)} dec_keys={len(dec_sd)}; "
            f"sample={next(iter(sd.keys()), None)!r}"
        )
    encoder.load_state_dict(enc_sd)
    decoder.load_state_dict(dec_sd)
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    return encoder, decoder, model_name
