"""CoolMeltWrapper: subclass of BigCodec's CodecLightningModule that inserts
the Melt (or, later, Cool) compression op between encoder output and quantizer
input. Stage is selected at construction by `cfg.train.compression.kind`.
"""
from itertools import chain
from typing import Optional

import torch.optim as optim

from lightning_module import CodecLightningModule
from common.schedulers import WarmupLR
from melt_manager import MeltManager


class CoolMeltWrapper(CodecLightningModule):
    def __init__(self, cfg,
                 melt_manager: Optional[MeltManager] = None,
                 cool_manager=None):
        super().__init__(cfg)
        self.melt_manager = melt_manager
        self.cool_manager = cool_manager
        self._compression_kind = cfg.train.compression.kind
        if cfg.train.get('freeze_encoder', False):
            for p in self.model['CodecEnc'].parameters():
                p.requires_grad = False

    def forward(self, batch):
        wav = batch['wav']
        vq_emb = self.model['CodecEnc'](wav.unsqueeze(1))

        if self.training:
            if self._compression_kind == 'melt':
                if self.melt_manager is None:
                    raise RuntimeError("compression.kind='melt' but no melt_manager was provided")
                vq_emb = self.melt_manager.melt(vq_emb, step=self.global_step)
            elif self._compression_kind == 'cool':
                if self.cool_manager is None:
                    raise RuntimeError("compression.kind='cool' but no cool_manager was provided")
                vq_emb = self.cool_manager.compress(vq_emb)
            else:
                raise ValueError(f"Unknown compression.kind: {self._compression_kind!r}")

        vq_post_emb, vq_code, vq_loss = self.model['generator'](vq_emb, vq=True)
        y_ = self.model['generator'](vq_post_emb, vq=False)
        return {
            'gt_wav': wav.unsqueeze(1),
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
        }

    def configure_optimizers(self):
        disc_params = list(self.model['discriminator'].parameters())
        if 'spec_discriminator' in self.model:
            disc_params = disc_params + list(self.model['spec_discriminator'].parameters())
        gen_params = [
            p for p in chain(self.model['CodecEnc'].parameters(),
                             self.model['generator'].parameters())
            if p.requires_grad
        ]

        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        return [gen_opt, disc_opt], [gen_sche, disc_sche]
