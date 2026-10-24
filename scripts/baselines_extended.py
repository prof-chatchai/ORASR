"""Extended baselines for the cost and safety comparison (v1.3.0).

Seed-42 cohort; high-alert type flags as in scripts/structured_misrouting.py.
Each monitor is evaluated with exact scores and under Gaussian score noise
(sigma = 0.10, 20 replicates). Parametric costs: c = 10 ms per gate and
A in {30, 300} s per approval (not a measurement).

Monitors
  flat4_no_approval        G1-G4 on every action, no approval
  flat4_universal_approval G1-G4 on every action, approval on every action
  flat4_risk_approval      G1-G4 on every action, approval iff rho_hat >= 0.7
  flat4_type_approval      G1-G4 on every action, approval iff high-alert type
  orasr                    tiered gates, approval on Safe pathway
  orasr_type_floor         ORASR with high-alert types forced Safe

Metrics
  gates_per_call, approvals_per_1000, expected_time_s (c, A)
  missed_approval_per_1000  truly-Safe actions (rho >= 0.7) executed without approval
  missed_hi_alert_approval_per_1000  high-alert actions executed without approval
  g23_skipped_on_true_normal_or_safe_pct  dose/allergy/interaction/duplicate checks
                            not applied to truly Normal-or-Safe actions
Output: results/baselines_extended.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from structured_misrouting import SEED, setup  # noqa: E402

C_MS = 10.0
APPROVAL_S = (30.0, 300.0)
MONITORS = ("flat4_no_approval", "flat4_universal_approval", "flat4_risk_approval",
            "flat4_type_approval", "orasr", "orasr_type_floor")


def evaluate(mon: str, obs, hi_alert, truth_safe, truth_ns) -> Dict[str, float]:
    n = len(obs)
    path = np.where(obs < 0.3, 0, np.where(obs < 0.7, 1, 2))
    if mon == "orasr_type_floor":
        path = np.where(hi_alert, 2, path)
    if mon.startswith("flat4"):
        gates = np.full(n, 4)
        g23 = np.ones(n, dtype=bool)
        approve = {"flat4_no_approval": np.zeros(n, bool), "flat4_universal_approval": np.ones(n, bool),
                   "flat4_risk_approval": obs >= 0.7, "flat4_type_approval": hi_alert}[mon]
    else:
        gates = np.array([1, 3, 4])[path]
        g23 = path >= 1
        approve = path == 2
    out = {"gates_per_call": float(gates.mean()),
           "approvals_per_1000": float(1000 * approve.mean()),
           "missed_approval_per_1000": float(1000 * np.mean(truth_safe & ~approve)),
           "missed_hi_alert_approval_per_1000": float(1000 * np.mean(hi_alert & ~approve)),
           "g23_skipped_on_true_normal_or_safe_pct": float(100 * np.mean(~g23[truth_ns]))}
    for a in APPROVAL_S:
        out[f"expected_time_s_A{int(a)}"] = float(gates.mean() * C_MS / 1000 + a * approve.mean())
    return out


def run() -> Dict:
    rho, strata, _sub, _enc, hi_alert = setup()
    truth_safe, truth_ns = strata == "SAFE", strata != "FAST"
    res: Dict[str, Dict] = {"exact": {}, "noise_sigma_0.10": {}}
    for mon in MONITORS:
        res["exact"][mon] = {k: round(v, 4) for k, v in
                             evaluate(mon, rho, hi_alert, truth_safe, truth_ns).items()}
        reps = [evaluate(mon, np.clip(rho + np.random.default_rng([SEED, r]).normal(0, .1, len(rho)), 0, 1),
                         hi_alert, truth_safe, truth_ns) for r in range(20)]
        res["noise_sigma_0.10"][mon] = {k: round(float(np.mean([x[k] for x in reps])), 4) for k in reps[0]}
    return {"seed": SEED, "gate_cost_ms": C_MS, "approval_wait_s": list(APPROVAL_S),
            "high_alert_share_of_all": round(float(hi_alert.mean()), 4), "results": res,
            "note": "parametric cost model; approval assumed to catch the hazard when requested"}


def main() -> None:
    out = run()
    p = ROOT / "results" / "baselines_extended.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for cond, d in out["results"].items():
        print(cond)
        for m, v in d.items():
            print("  ", m, v)


if __name__ == "__main__":
    main()
