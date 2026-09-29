"""Shared pytest fixtures for the modbus_replay_simulator test suite.

Installs a minimal ``torch`` / ``torch.nn`` stub into ``sys.modules``
before any test collection runs, so that ``modbus_replay.ids_worker_torch``
(which does ``import torch.nn as nn`` at module load) imports cleanly on
machines without torch installed. Tests that need a working fake model
use the ``fake_torch`` fixture in ``tests/test_ids_worker.py`` to patch
the model constructor.

The stub needs to expose the bits that ``ids_worker_torch`` uses at
*import time* (before tests run):

* ``torch.nn.Module`` — base class for SEBlock / TCNBlockSE / TCNClassifierSE
* ``torch.nn.Conv1d``, ``BatchNorm1d``, ``Dropout``, ``Linear``,
  ``AdaptiveAvgPool1d`` — referenced inside the class bodies but not
  actually executed (the test monkeypatches ``build_tcn_se_model`` to
  return a constant-logit fake).

We give them all a no-op ``__init__`` so the bodies don't crash if
something accidentally instantiates them.
"""
from __future__ import annotations

import sys
import types


def _make_noop_class(name: str) -> type:
    cls = type(
        name,
        (),
        {
            "__init__": lambda self, *a, **kw: None,
            "__call__": lambda self, *a, **kw: self,
            "load_state_dict": lambda self, *a, **kw: None,
            "eval": lambda self, *a, **kw: None,
            "state_dict": lambda self, *a, **kw: {},
        },
    )
    return cls


if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__path__ = []  # marks it as a package
    # Provide a minimal ``torch.load`` that returns an empty
    # state_dict. Tests that need load to fail (e.g. the bad-path
    # test) override this via monkeypatch.
    def _stub_load(path, map_location=None, weights_only=False):
        return {"state_dict": {}}
    _fake_torch.load = _stub_load
    # Some libraries (e.g. scipy's array_api compat layer) look up
    # ``torch.Tensor`` defensively. Provide an attribute so that lookup
    # returns something usable rather than raising AttributeError.
    class _StubTensor:
        pass
    _fake_torch.Tensor = _StubTensor
    sys.modules["torch"] = _fake_torch
if "torch.nn" not in sys.modules:
    _fake_nn = types.ModuleType("torch.nn")
    _fake_nn.__path__ = []
    _fake_nn.Module = _make_noop_class("Module")
    _fake_nn.Conv1d = _make_noop_class("Conv1d")
    _fake_nn.BatchNorm1d = _make_noop_class("BatchNorm1d")
    _fake_nn.Dropout = _make_noop_class("Dropout")
    _fake_nn.Linear = _make_noop_class("Linear")
    _fake_nn.AdaptiveAvgPool1d = _make_noop_class("AdaptiveAvgPool1d")
    sys.modules["torch.nn"] = _fake_nn