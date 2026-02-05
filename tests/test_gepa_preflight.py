import logging
from pathlib import Path

from experiments.gepa import run_gepa


def test_load_dataset_missing(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    missing = tmp_path / "nope.jsonl"
    res = run_gepa._load_dataset(missing)
    assert res == []
    assert any("Dataset not found" in rec.message for rec in caplog.records)