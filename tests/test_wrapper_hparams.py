"""Unit tests for the hparam strip that keeps Lightning checkpoints picklable.

The full ``CoolMeltWrapper`` requires the BigCodec model graph and is
covered by ``cool_smoke.py``. Here we test the strip helper directly with
a dummy holder that mimics Lightning's hparam containers.
"""
import sys

sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final")
sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final/external/BigCodec")

import pickle

import pytest

from cool_manager import CoolManager
from melt_manager import MeltManager
from melt_wrapper import _strip_runtime_hparams


class _Holder:
    pass


def test_strip_removes_managers_from_dict_hparams():
    h = _Holder()
    cm = CoolManager(n_jobs=4)
    mm = MeltManager()
    h._hparams = {"cfg": "stub", "melt_manager": mm, "cool_manager": cm}
    h._hparams_initial = {"cfg": "stub", "melt_manager": mm, "cool_manager": cm}
    _strip_runtime_hparams(h)
    assert h._hparams == {"cfg": "stub"}
    assert h._hparams_initial == {"cfg": "stub"}


def test_strip_is_noop_when_keys_absent():
    h = _Holder()
    h._hparams = {"cfg": "stub"}
    _strip_runtime_hparams(h)
    assert h._hparams == {"cfg": "stub"}


def test_strip_tolerates_missing_hparams_attr():
    h = _Holder()
    _strip_runtime_hparams(h)  # must not raise


def test_hparams_pickle_after_strip_with_live_pool():
    """The whole point: after strip, hparams pickle even if the original
    CoolManager has its joblib pool alive.
    """
    import torch
    cm = CoolManager(n_jobs=4)
    cm.compress(torch.randn(2, 3, 8))
    assert cm._pool is not None

    h = _Holder()
    h._hparams = {"cfg": "stub", "cool_manager": cm}
    h._hparams_initial = {"cfg": "stub", "cool_manager": cm}
    with pytest.raises(TypeError):
        pickle.dumps(h._hparams)

    _strip_runtime_hparams(h)
    pickle.dumps(h._hparams)
    pickle.dumps(h._hparams_initial)
