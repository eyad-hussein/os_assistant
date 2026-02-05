import importlib
import sys
from pathlib import Path

import pytest

from experiments.gepa import gepa_adapter


def test_is_gepa_available_false(monkeypatch):
    # Ensure no dspy in sys.modules
    monkeypatch.setitem(sys.modules, 'dspy', None)
    assert not gepa_adapter.is_gepa_available()


def test_run_gepa_experiment_calls_gepa(monkeypatch, tmp_path):
    # Create a fake dspy module with the common GEPA API
    class FakeExp:
        def __init__(self, **kwargs):
            self.kw = kwargs

        def run(self):
            return {"ok": True, "params": self.kw}

    fake_gepa = type("GEPAModule", (), {"Experiment": FakeExp})

    # Insert fake dspy into sys.modules
    fake_dspy = type("Dspy", (), {"gepa": fake_gepa})
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    # Run experiment
    cfg = {"dataset_path": "dummy", "variants_dir": "dummy_variants"}
    res = gepa_adapter.run_gepa_experiment(cfg)

    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert "_results_path" in res

    # Check result file exists
    assert Path(res["_results_path"]).exists()


@pytest.mark.skipif('dspy' in sys.modules, reason="Requires mocking or no real dspy installed")
def test_run_gepa_experiment_raises_on_missing_dspy(monkeypatch):
    # Ensure dspy not present
    monkeypatch.setitem(sys.modules, 'dspy', None)
    with pytest.raises(RuntimeError):
        gepa_adapter.run_gepa_experiment({})
