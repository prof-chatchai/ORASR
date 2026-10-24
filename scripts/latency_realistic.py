"""Routing latency with I/O-bound gates under concurrency (v1.3.0).

Each gate wraps the committed validator and first waits for a simulated
external service, with latencies drawn (seed 42) from lognormal distributions:
    G1 local schema check   no wait
    G2 formulary / dose KB  median 2 ms, log-SD 0.5
    G3 interaction service  median 5 ms, log-SD 0.6
    G4 output validation    median 1 ms, log-SD 0.5
1,000 requests drawn from the seed-42 cohort mix are issued through a thread
pool at concurrency 1, 2, 4, 8, 16, 32, 64, for ORASR and for Flat-4 (every
action on the Safe gate stack). Reported: end-to-end latency p50/p95/p99 and
throughput. A second scenario gives the G3 service a capacity of 8 concurrent
calls (semaphore), so requests queue for it. Values are host-specific (environment block included); the
simulated service times are an assumption, not a measurement of real services.
Output: results/latency_realistic.json
"""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402
from orasr import GateType, ORASRRouter, SafetyGate  # noqa: E402

logging.disable(logging.CRITICAL)
SEED = lb.SEED
N_REQ = 1000
LEVELS = (1, 2, 4, 8, 16, 32, 64)
SERVICE = {GateType.RISK_ASSESSMENT: (0.002, 0.5), GateType.CONSTRAINT_VALIDATION: (0.005, 0.6),
           GateType.POSTCONDITION: (0.001, 0.5)}


def build(waits: Dict[int, Dict[GateType, float]], g3_capacity: int = 0) -> ORASRRouter:
    r = ORASRRouter(enable_audit=False)
    sem = threading.Semaphore(g3_capacity) if g3_capacity else None
    for gt, gate in list(r.gates.items()):
        if gt not in SERVICE:
            continue
        inner = gate.validator

        def v(ctx, _inner=inner, _gt=gt):
            if sem is not None and _gt is GateType.CONSTRAINT_VALIDATION:
                with sem:  # shared service with limited capacity -> queueing
                    time.sleep(waits[ctx["req_id"]][_gt])
            else:
                time.sleep(waits[ctx["req_id"]][_gt])
            return _inner(ctx)
        r.gates[gt] = SafetyGate(gate_type=gt, validator=v, name=gate.name, description=gate.description)
    return r


def run_level(router, reqs: List[Dict], conc: int, force_safe: bool) -> Dict[str, float]:
    def one(q):
        t0 = time.perf_counter()
        router.route(action=lambda d: d, input_data={"a": 1}, risk_score=q["risk"], human_approved=True,
                     require_human_approval=force_safe, req_id=q["id"])
        return time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        lat = list(ex.map(one, reqs))
    wall = time.perf_counter() - t0
    a = np.array(lat) * 1000
    return {"concurrency": conc, "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3),
            "mean_ms": round(float(a.mean()), 3), "throughput_rps": round(len(reqs) / wall, 1)}


def run() -> Dict:
    cohort = lb.build_cohort(random.Random(SEED))
    rng = random.Random(SEED)
    reqs = [{"id": i, "risk": c["risk"]} for i, c in enumerate(rng.sample(cohort, N_REQ))]
    nrng = np.random.default_rng(SEED)
    waits = {q["id"]: {gt: float(nrng.lognormal(np.log(m), s)) for gt, (m, s) in SERVICE.items()}
             for q in reqs}
    out = {}
    for scen, cap in (("unlimited_services", 0), ("g3_capacity_8", 8)):
        router = build(waits, cap)
        for q in reqs[:50]:  # warm-up
            router.route(action=lambda d: d, input_data={"a": 1}, risk_score=q["risk"],
                         human_approved=True, req_id=q["id"])
        out[scen] = {"orasr": [], "flat4": []}
        for conc in LEVELS:
            out[scen]["orasr"].append(run_level(router, reqs, conc, False))
            out[scen]["flat4"].append(run_level(router, reqs, conc, True))
    mean_service = {gt.value: round(1000 * float(np.mean([w[gt] for w in waits.values()])), 3)
                    for gt in SERVICE}
    return {"seed": SEED, "n_requests": N_REQ, "service_model": {gt.value: {"median_s": m, "log_sd": s}
                                                                 for gt, (m, s) in SERVICE.items()},
            "realised_mean_service_ms": mean_service, "results": out,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "processor": platform.processor()},
            "note": "flat4 = require_human_approval=True forces the Safe gate stack for every request"}


def main() -> None:
    out = run()
    p = ROOT / "results" / "latency_realistic.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for scen, d in out["results"].items():
        for k, rows in d.items():
            for r in rows:
                print(scen, k, r)


if __name__ == "__main__":
    main()
