import json
from pathlib import Path
import os

import pytest

from experiments.gepa.benchmark import BenchmarkRunner


class FakeAssistant:
    def __init__(self, *args, **kwargs):
        self.config = {"configurable": {"thread_id": "fake"}}
        self.invoked = []

    def process_prompt(self, prompt, image=None, initial_state=None):
        # Behavior depends on PROMPT_VARIANT env var
        variant = os.getenv("PROMPT_VARIANT")
        # Create fake final_result-like object (simple namespace)
        class Resp:
            pass

        class FinalResult:
            def __init__(self, response_type, command=None, answer=None, tool_exec=None):
                self.response_type = response_type
                if response_type == "command":
                    r = Resp()
                    r.command = command
                    r.tool_execution = tool_exec
                    self.response = r
                else:
                    r = Resp()
                    r.answer = answer
                    r.tool_execution = tool_exec
                    self.response = r

        # Simplified decision: if variant == 'strict_json' succeed, else fallback
        if variant == "strict_json":
            self._final_result = FinalResult("command", command="du -ah /var/log | sort -rh | head -n 20")
        else:
            # fallback response
            self._final_result = FinalResult("information", answer="Unable to generate an answer")

    # mimic the graph state retrieval
    @property
    def app(self):
        class _App:
            def __init__(self, final_result):
                self._final_result = final_result

            def get_state(self, config=None):
                class S:
                    def __init__(self, v):
                        self.values = {"final_result": v}

                return S(self._final_result)

        return _App(self._final_result)


@pytest.fixture(autouse=True)
def patch_osassistant(monkeypatch):
    import os_assistant.os_assistant as osa

    monkeypatch.setattr(osa, "OSAssistant", FakeAssistant)


def test_benchmark_runs_and_saves(tmp_path, monkeypatch):
    ds = tmp_path / "data.jsonl"
    ds.write_text(json.dumps({"id": "cmd1", "node": "command_generation_node", "prompt": "Find largest files in /var/log", "expected_type": "command", "expect_contains": "du -ah"}) + "\n")

    runner = BenchmarkRunner(dataset_path=ds, results_dir=tmp_path)

    res = runner.run_head_to_head(variants=["strict_json"])  # runs baseline + strict_json

    assert "baseline" in res and "strict_json" in res
    # Baseline should be worse (fallback -> information)
    assert res["baseline"]["type_accuracy"] == 0
    assert res["strict_json"]["type_accuracy"] == 1

    # Also test repeats/seed support
    res2 = runner.run_head_to_head(variants=["strict_json"], repeats=3, seed=42)
    assert "baseline" in res2 and "strict_json" in res2
    # The repeated-run results include repeats/seed metadata
    assert res2["strict_json"].get("repeats") == 3
    assert res2["strict_json"].get("seed") == 42

    # Results file exists
    assert Path(res["_results_path"]).exists()
