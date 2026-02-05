import json
import os
from pathlib import Path

from experiments.gepa.benchmark import BenchmarkRunner


class FakeEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # create a fake dataset with one sample
        self.dataset = type("D", (), {"samples": [type("S", (), {"model_dump": lambda self: {"question": "Q", "expected_response": "A", "type": "command", "domain": "file_system"}})()]})

    def load_dataset(self, path=None):
        return self.dataset

    def evaluate_sample(self, sample):
        # return a deterministic evaluation result
        return {"query": sample.get("question"), "overall_score": 4.0}


def test_run_head_to_head_with_evaluator(monkeypatch, tmp_path):
    # Make a small dataset file used by BenchmarkRunner but evaluator uses its own dataset
    ds = tmp_path / "data.jsonl"
    ds.write_text(json.dumps({"id": "cmd1", "node": "command_generation_node", "prompt": "Find largest files in /var/log", "expected_type": "command", "expect_contains": "du -ah"}) + "\n")

    runner = BenchmarkRunner(dataset_path=ds, results_dir=tmp_path)

    # Patch OSAssistantEvaluator to our fake evaluator
    import evaluator.core.evaluator as ev

    monkeypatch.setattr(ev, "OSAssistantEvaluator", FakeEvaluator)

    res = runner.run_head_to_head(variants=["strict_json"], use_evaluator=True)

    assert "baseline" in res and "strict_json" in res
    assert res["baseline"]["num_examples"] == 1
    assert res["strict_json"]["num_examples"] == 1
    assert Path(res["_results_path"]).exists()
    # Clean up PROMPT_VARIANT if set
    if "PROMPT_VARIANT" in os.environ:
        del os.environ["PROMPT_VARIANT"]
