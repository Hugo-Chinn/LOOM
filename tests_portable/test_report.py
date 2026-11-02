import json
import pytest
from loom.report import summarize


def test_report_pairs_only_shared_seeds(tmp_path):
    for variant, seed, auc in [("full", 27, .8), ("full", 28, .9),
                               ("no_bcu", 27, .7), ("no_bcu", 29, .95)]:
        out = tmp_path / variant / f"S{seed}"
        out.mkdir(parents=True)
        (out / "metrics.json").write_text(json.dumps({
            "seed": seed, "test": {"buy": {"auc": auc, "logloss": .5, "observed": 100}},
        }))
    rows = summarize(tmp_path)
    candidate = next(r for r in rows if r["variant"] == "no_bcu")
    assert candidate["n_seeds"] == 2
    assert candidate["n_paired"] == 1
    assert candidate["paired_delta_mean"] == pytest.approx(-.1)
    assert candidate["paired_delta_sample_std"] is None
