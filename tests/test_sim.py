import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from cplan import SimConfig, SchedulerSim, arrival_poisson  # noqa: E402


def _load(n=120, seed=0, slo=3000.0):
    rng = np.random.default_rng(seed)
    reqs = arrival_poisson(n, rate_per_s=1.5, rng=rng, slo_ms=slo)
    return reqs


def test_continuous_beats_static_on_ttft():
    reqs_a, reqs_b = _load(), _load()
    cont = SchedulerSim(SimConfig(), "continuous").run(reqs_a)
    stat = SchedulerSim(SimConfig(), "static").run(reqs_b)
    assert cont["mean_ttft_ms"] < stat["mean_ttft_ms"] * 0.7, (cont, stat)


def test_continuous_beats_static_on_p99():
    reqs_a, reqs_b = _load(), _load()
    cont = SchedulerSim(SimConfig(), "continuous").run(reqs_a)
    stat = SchedulerSim(SimConfig(), "static").run(reqs_b)
    assert cont["p99_end_ms"] < stat["p99_end_ms"], (cont, stat)


def test_slo_attainment_higher():
    # SLO 1500ms sits between the two modes' p99 (continuous ~1.5s, static ~2.3s)
    reqs_a, reqs_b = _load(slo=1500.0), _load(slo=1500.0)
    cont = SchedulerSim(SimConfig(), "continuous").run(reqs_a)
    stat = SchedulerSim(SimConfig(), "static").run(reqs_b)
    assert cont["slo_attainment"] > stat["slo_attainment"], (cont, stat)


def test_all_requests_complete_both_modes():
    for mode in ("continuous", "static"):
        reqs = _load(n=40)
        res = SchedulerSim(SimConfig(), mode).run(reqs)
        assert res["completed"] == 40, (mode, res)
        assert all(r.end_ms is not None for r in reqs)


def test_higher_slot_count_improves_ttft():
    rng = np.random.default_rng(3)
    base = arrival_poisson(80, 1.5, rng)
    import copy

    r1 = copy.deepcopy(base)
    r2 = copy.deepcopy(base)
    small = SchedulerSim(SimConfig(batch_slots=8), "continuous").run(r1)
    big = SchedulerSim(SimConfig(batch_slots=32), "continuous").run(r2)
    assert big["mean_ttft_ms"] < small["mean_ttft_ms"], (small, big)


def test_chunked_prefill_beats_all_at_once_ttft():
    """Chunking the prefill of admitted requests moves first tokens earlier."""
    import copy

    rng = np.random.default_rng(5)
    base = arrival_poisson(60, 2.0, rng, max_prompt=2048)
    r1 = copy.deepcopy(base)
    r2 = copy.deepcopy(base)
    whole = SchedulerSim(SimConfig(chunk_tokens=4096), "continuous").run(r1)
    chunked = SchedulerSim(SimConfig(chunk_tokens=256), "continuous").run(r2)
    assert chunked["mean_ttft_ms"] < whole["mean_ttft_ms"], (whole, chunked)
