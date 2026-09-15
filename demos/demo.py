"""Demo: continuous vs static batching sweep + receipts + chart."""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cplan import SimConfig, SchedulerSim, arrival_poisson  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

sweep = {}
for rate in (0.8, 1.5, 2.5, 4.0):
    rng = np.random.default_rng(42)
    base = arrival_poisson(100, rate, rng, slo_ms=4000.0)
    r_cont, r_stat = copy.deepcopy(base), copy.deepcopy(base)
    cont = SchedulerSim(SimConfig(), "continuous").run(r_cont)
    stat = SchedulerSim(SimConfig(), "static").run(r_stat)
    sweep[f"rate={rate}/s"] = {
        "continuous": {"ttft": cont["mean_ttft_ms"], "p99": cont["p99_end_ms"], "slo": cont["slo_attainment"]},
        "static": {"ttft": stat["mean_ttft_ms"], "p99": stat["p99_end_ms"], "slo": stat["slo_attainment"]},
        "ttft_improvement": f"{(1 - cont['mean_ttft_ms'] / stat['mean_ttft_ms']):.0%}",
    }

receipt = {"sweep": sweep}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = [0.8, 1.5, 2.5, 4.0]
    cont_ttft = [sweep[f"rate={r}/s"]["continuous"]["ttft"] for r in rates]
    stat_ttft = [sweep[f"rate={r}/s"]["static"]["ttft"] for r in rates]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rates, stat_ttft, "o-", label="static batching")
    ax1.plot(rates, cont_ttft, "s-", label="continuous batching")
    ax1.set_xlabel("arrival rate (req/s)")
    ax1.set_ylabel("mean TTFT (ms)")
    ax1.set_title("Time-to-first-token under load")
    ax1.legend()
    cont_slo = [sweep[f"rate={r}/s"]["continuous"]["slo"] for r in rates]
    stat_slo = [sweep[f"rate={r}/s"]["static"]["slo"] for r in rates]
    ax2.plot(rates, stat_slo, "o-", label="static")
    ax2.plot(rates, cont_slo, "s-", label="continuous")
    ax2.set_xlabel("arrival rate (req/s)")
    ax2.set_ylabel("SLO attainment")
    ax2.set_title("4s-SLO attainment")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batching.png"), dpi=140)
    print("chart -> receipts/batching.png")
except Exception as e:  # pragma: no cover
    print("chart skipped:", e)

for k, v in sweep.items():
    assert v["continuous"]["ttft"] < v["static"]["ttft"], k
    assert v["continuous"]["slo"] >= v["static"]["slo"], k
print("DEMO_OK")
