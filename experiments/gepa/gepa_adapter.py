from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def is_gepa_available() -> bool:
    try:
        import dspy

        return True
    except Exception:
        return False


def _save_results(results: dict, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir or Path("experiments") / "gepa" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"gepa_result_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_file


def run_gepa_experiment(config: dict) -> dict:
    """Run a GEPA experiment using dspy.

    The function supports a few plausible dspy API shapes to remain robust
    across versions. It returns the results dictionary as produced by GEPA and
    also saves it under `experiments/gepa/results/` for auditing.

    Args:
        config: A dictionary describing the experiment. Expected keys include
                'dataset_path', 'variants_dir', 'metrics', and other GEPA options.

    Returns:
        Results dict (may be forward-compatible depending on dspy's API).

    Raises:
        RuntimeError with helpful messages if dspy isn't installed or if the
        GEPA API is not recognized.
    """
    if not is_gepa_available():
        raise RuntimeError(
            "dspy/GEPA is not installed. Install it with `pip install dspy` to run experiments."
        )

    # Lazy import inside function to avoid hard dependency at module import time
    import dspy  # type: ignore

    # Try gepa namespace first
    try:
        if hasattr(dspy, "gepa"):
            ge = getattr(dspy, "gepa")
            # If there is a high-level Experiment class
            if hasattr(ge, "Experiment"):
                LOGGER.info("Using dspy.gepa.Experiment API")
                exp = ge.Experiment(**config)
                res = exp.run()
            # Or a run_experiment function
            elif hasattr(ge, "run_experiment"):
                LOGGER.info("Using dspy.gepa.run_experiment API")
                res = ge.run_experiment(config)
            else:
                raise RuntimeError("dspy.gepa API not recognized; please update adapter.")
        # Try top-level run_experiment (older/newer variants)
        elif hasattr(dspy, "run_experiment"):
            LOGGER.info("Using dspy.run_experiment API")
            res = dspy.run_experiment(config)
        elif hasattr(dspy, "experiments") and hasattr(dspy.experiments, "run"):
            LOGGER.info("Using dspy.experiments.run API")
            res = dspy.experiments.run(config)
        else:
            raise RuntimeError("dspy installed but no GEPA entrypoint found; update the adapter.")

    except Exception as e:
        LOGGER.exception("GEPA experiment failed")
        raise RuntimeError(f"GEPA experiment failed: {e}")

    # Ensure we have a dict-like result
    if not isinstance(res, dict):
        try:
            res = dict(res)
        except Exception:
            res = {"result": str(res)}

    # Save the results for traceability
    out_path = _save_results(res, config.get("results_dir"))
    res["_results_path"] = str(out_path)

    LOGGER.info(f"GEPA experiment finished; results saved to {out_path}")

    return res
