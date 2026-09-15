"""Continuous-batching scheduler simulation.

Two schedulers, identical engine model, identical arrivals:

  * static batching — the "batch-and-wait" server: a batch is formed, run
    to completion (every request pays the longest request's length), then
    the next batch forms.
  * continuous batching (prior art: Orca, Yu et al. 2021; vLLM) — the
    iteration-level scheduler: a slot frees at each decode step and is
    immediately refilled from the queue.

All latencies are ARRIVAL-RELATIVE (end_ms = t - r.arrive_s in ms): a
request that arrives into an idle server and finishes one iteration later
has latency ~= one iteration, not absolute wall clock.

Engine model (per iteration):
    prefill:  compute-bound, ~linear in tokens (ms per 1k)
    decode:   memory-bound, batch-amortized (near-constant per iteration
              until slots saturate)
SLO attainment = share of requests with end-to-end latency <= slo_ms.

This is a simulator, and says so: absolute numbers are only as good as
the per-iteration cost model fed in. The *comparison* — continuous vs
static under the same model — is the product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Request:
    rid: int
    arrive_s: float
    prompt_len: int
    output_len: int
    slo_ms: float
    first_tok_ms: float | None = None
    end_ms: float | None = None


@dataclass
class SimConfig:
    batch_slots: int = 32
    prefill_ms_per_1k: float = 60.0
    decode_ms_per_iter: float = 25.0
    chunk_tokens: int = 512


def arrival_poisson(n: int, rate_per_s: float, rng: np.random.Generator,
                    max_prompt: int = 512, out_range: tuple[int, int] = (16, 128),
                    slo_ms: float = 2500.0) -> list[Request]:
    gaps = rng.exponential(1 / rate_per_s, n)
    t = np.cumsum(gaps)
    reqs = []
    for i, (ti, pl, ol) in enumerate(zip(t, rng.integers(64, max_prompt, n),
                                         rng.integers(*out_range, n))):
        reqs.append(Request(i, float(ti), int(pl), int(ol), slo_ms))
    return reqs


class SchedulerSim:
    def __init__(self, cfg: SimConfig, mode: str = "continuous"):
        assert mode in ("continuous", "static")
        self.cfg = cfg
        self.mode = mode

    def run(self, requests: list[Request]) -> dict:
        if self.mode == "continuous":
            return self._run_continuous(requests)
        return self._run_static(requests)

    def _prefill_ms(self, tokens: int) -> float:
        return self.cfg.prefill_ms_per_1k * tokens / 1000

    def _decode_iter_ms(self, batch_size: int) -> float:
        return self.cfg.decode_ms_per_iter * (0.4 + 0.6 * batch_size / self.cfg.batch_slots)

    # ----------------------------------------------------------- static
    def _run_static(self, requests: list[Request]) -> dict:
        t = 0.0
        pending = sorted(requests, key=lambda r: r.arrive_s)
        i = 0
        while i < len(pending):
            batch = []
            while i < len(pending) and len(batch) < self.cfg.batch_slots \
                    and pending[i].arrive_s <= t:
                batch.append(pending[i])
                i += 1
            if not batch:
                t = pending[i].arrive_s
                continue
            t += self._prefill_ms(sum(r.prompt_len for r in batch)) / 1000
            # decode to the longest output; per-request completion at its own
            # output length (a finished request still occupies its slot —
            # that head-of-line blocking is the static-batching pathology)
            elapsed = 0.0
            longest = max(r.output_len for r in batch)
            for step in range(1, longest + 1):
                elapsed += self._decode_iter_ms(len(batch)) / 1000
                for r in batch:
                    if step == 1 and r.first_tok_ms is None:
                        r.first_tok_ms = (t + elapsed - r.arrive_s) * 1000
                    if step == r.output_len:
                        r.end_ms = (t + elapsed - r.arrive_s) * 1000
            t += elapsed
        return self._stats(requests)

    # ------------------------------------------------------ continuous
    def _run_continuous(self, requests: list[Request]) -> dict:
        t = 0.0
        queue = sorted(requests, key=lambda r: r.arrive_s)
        qi = 0
        active: list[Request] = []
        state: dict[int, dict] = {}
        done = 0
        n = len(requests)
        while done < n:
            while qi < n and len(active) < self.cfg.batch_slots \
                    and queue[qi].arrive_s <= t:
                r = queue[qi]
                qi += 1
                state[r.rid] = {"prefill_left": r.prompt_len, "decoded": 0}
                active.append(r)
            if not active:
                t = queue[qi].arrive_s
                continue
            pre = [r for r in active if state[r.rid]["prefill_left"] > 0]
            if pre:
                chunk = self.cfg.chunk_tokens * len(pre)
                t += self._prefill_ms(chunk) / 1000
                for r in pre:
                    st = state[r.rid]
                    st["prefill_left"] = max(0, st["prefill_left"] - self.cfg.chunk_tokens)
                    if st["prefill_left"] == 0:
                        r.first_tok_ms = (t - r.arrive_s) * 1000
                continue
            t += self._decode_iter_ms(len(active)) / 1000
            for r in list(active):
                st = state[r.rid]
                st["decoded"] += 1
                if st["decoded"] >= r.output_len:
                    r.end_ms = (t - r.arrive_s) * 1000
                    active.remove(r)
                    done += 1
        return self._stats(requests)

    def _stats(self, requests: list[Request]) -> dict:
        ends = [r.end_ms for r in requests if r.end_ms is not None]
        ttfts = [r.first_tok_ms for r in requests if r.first_tok_ms is not None]
        slo = requests[0].slo_ms if requests else 1000.0
        met = sum(1 for r in requests if r.end_ms is not None and r.end_ms <= slo)
        return {
            "mode": self.mode,
            "completed": len(ends),
            "p50_end_ms": float(np.percentile(ends, 50)) if ends else None,
            "p99_end_ms": float(np.percentile(ends, 99)) if ends else None,
            "mean_ttft_ms": float(np.mean(ttfts)) if ttfts else None,
            "slo_attainment": round(met / len(requests), 4),
        }
