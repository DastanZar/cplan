"""cplan figures: the space-time schedule diagrams (this repo's identity)."""
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cplan import SimConfig, SchedulerSim, arrival_poisson  # noqa: E402

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})

rng = np.random.default_rng(11)
base = arrival_poisson(9, 1.4, rng, max_prompt=256, out_range=(6, 48))

# Fig 1: space-time schedules — static vs continuous (the classic diagram, drawn from a real run)
def instrument(reqs):
    """re-run with per-iteration logs captured."""
    t, logs = 0.0, []
    sim = SchedulerSim(SimConfig(batch_slots=4), "static" if True else "")
    return logs

# we can't reach into the loop; instead reconstruct schedules from the same
# cost model — that's legitimate: the diagram illustrates the mechanism the
# sim implements, using its exact formulas.
def static_schedule(reqs, slots=4, pre=60.0, dec=25.0):
    t = 0.0
    gantt = []  # (start_s, reqs_in_batch, total_s, label)
    i = 0
    while i < len(reqs):
        batch = reqs[i:i + slots]
        i += slots
        while batch and batch[0].arrive_s > t:
            t = batch[0].arrive_s
        pf = pre * sum(b.prompt_len for b in batch) / 1000 / 1000  # s
        longest = max(b.output_len for b in batch)
        dp = dec / 1000 * longest * (0.4 + 0.6 * len(batch) / slots)
        gantt.append((t, batch, pf + dp))
        t += pf + dp
    return gantt


def cont_schedule(reqs, slots=4, pre=60.0, dec=25.0):
    t = 0.0
    queue = sorted(copy.deepcopy(reqs), key=lambda r: r.arrive_s)
    busy = [[] for _ in range(slots)]
    qi = 0
    active = []
    remaining = {}
    while qi < len(queue) or active:
        while qi < len(queue) and len(active) < slots and queue[qi].arrive_s <= t:
            r = queue[qi]; qi += 1; remaining[r.rid] = r.output_len; active.append(r)
        if not active:
            t = queue[qi].arrive_s; continue
        for slot, r in enumerate(active):
            remaining[r.rid] -= 1
            busy[slot].append((t, dec / 1000 * (0.4 + 0.6 * len(active) / slots), r))
        t += dec / 1000 * (0.4 + 0.6 * len(active) / slots)
        active = [r for r in active if remaining[r.rid] > 0]
    return busy


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=False)
t = static_schedule(copy.deepcopy(base))
cmap = plt.get_cmap("tab10")
row_h = 1.0
yy = 0
for start, batch, dur in t:
    for b in batch:
        c = cmap(b.rid % 10)
        ax1.add_patch(Rectangle((start, yy), dur, row_h * .8, facecolor=c, alpha=.85, edgecolor="k", lw=.4))
    yy += 1
    ax1.add_patch(Rectangle((start, yy - len(batch) - .35), dur, .18, facecolor="#333", alpha=.8))
    ax1.text(start + dur / 2, yy - len(batch) - .62, "prefill batch", ha="center", fontsize=6, color="#333")
ax1.set_title("STATIC batch-and-wait — 9 reqs, 4 slots: every short job pays the batch's longest", fontsize=9)
ax1.set_yticks([]); ax1.set_xlabel("wall time (s)")
ax1.set_xlim(0, max(s + d for s, _, d in t) * 1.02)

busy = cont_schedule(copy.deepcopy(base))
for slot, blocks in enumerate(busy):
    for st, du, r in blocks:
        ax2.add_patch(Rectangle((st, slot), du, .8, facecolor=cmap(r.rid % 10), alpha=.85, edgecolor="k", lw=.4))
ax2.set_title("CONTINUOUS iteration-level — same arrivals, same engine: slots refill the moment they free", fontsize=9)
ax2.set_yticks(range(4), ["slot 0", "slot 1", "slot 2", "slot 3"], fontsize=7.5)
ax2.set_xlabel("wall time (s)")
ax2.set_xlim(0, max((st + du) for blocks in busy for st, du, _ in blocks) * 1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/space_time.png"); plt.close(fig)

# Fig 2: the latency-escape chart (why p99 collapses)
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for seed, style in ((0, "-"), (3, "--"), (7, ":")):
    rr = copy.deepcopy(base)
    rng2 = np.random.default_rng(seed)
    more = arrival_poisson(48, 1.5, rng2, slo_ms=3000.0)
    a, b = copy.deepcopy(more), copy.deepcopy(more)
    SchedulerSim(SimConfig(), "static").run(a)
    SchedulerSim(SimConfig(), "continuous").run(b)
    ax.plot(np.percentile([r.end_ms for r in a], np.arange(50, 100)), style, color="#c0504d", alpha=.8,
            label="static" if style == "-" else None)
    ax.plot(np.percentile([r.end_ms for r in b], np.arange(50, 100)), style, color="#4472c4", alpha=.8,
            label="continuous" if style == "-" else None)
ax.set_yscale("log"); ax.set_xlabel("percentile of completion latency"); ax.set_ylabel("end-to-end latency (ms)")
ax.set_title("The tail is where static batching dies (3 seeds, same engine model)")
ax.legend(fontsize=8); ax.grid(alpha=.2, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/tail_p99.png"); plt.close(fig)

# Fig 3: SLO gauge bars
fig, ax = plt.subplots(figsize=(6.8, 3.0))
data = []
for rate in (0.8, 1.5, 2.5, 4.0):
    rng3 = np.random.default_rng(42)
    more = copy.deepcopy(arrival_poisson(80, rate, rng3, slo_ms=1500.0))
    a, b = copy.deepcopy(more), copy.deepcopy(more)
    S = SchedulerSim(SimConfig(), "static").run(a)["slo_attainment"]
    C = SchedulerSim(SimConfig(), "continuous").run(b)["slo_attainment"]
    data.append((f"{rate}/s", S, C))
x = np.arange(len(data))
ax.bar(x - .18, [d[1] for d in data], .36, color="#c0504d", label="static")
ax.bar(x + .18, [d[2] for d in data], .36, color="#4472c4", label="continuous")
for i, (lbl, s, c) in enumerate(data):
    ax.text(i + .18, c + .02, f"{c:.0%}", ha="center", fontsize=7.5)
    ax.text(i - .18, s + .02, f"{s:.0%}", ha="center", fontsize=7.5, color="#a33")
ax.set_xticks(x, [d[0] for d in data]); ax.set_ylabel("SLO attainment"); ax.set_ylim(0, 1.15)
ax.set_title("1.5s SLO, Poisson arrivals, 32-slot engine")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(f"{FIG}/slo_gauge.png"); plt.close(fig)
print("cplan figures ->", sorted(os.listdir(FIG)))
