"""Unreliable upstream risk scores (v1.3.0): fail-closed inputs, calibration
drift and prevalence shift. Deterministic, seed 42.

Part A  fail-closed sweep through the real router: 1,000 calls for each kind of
        unusable score (and for unusable uncertainty values). Every call must
        route Safe, be blocked, and record the R0_RiskInput violation.
Part B  calibration drift: rho_hat = expit(a * logit(rho) + b) on the seed-42
        cohort, a in {0.6, 0.8, 1.0, 1.25}, b in {-1, -0.5, 0, 0.5}; no noise.
Part C  prevalence shift: Safe-stratum share 10-40% (Fast:Normal kept 35:45)
        with Gaussian noise sigma = 0.10, 20 replicates; approvals and missed
        approvals per 1,000 actions, with and without a 0.60 guard band.
Output: results/upstream_reliability.json
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402
from orasr import ORASRRouter, ReasoningPath  # noqa: E402

logging.disable(logging.CRITICAL)
SEED = lb.SEED


def part_a() -> Dict:
    router = ORASRRouter(enable_audit=False)
    kinds = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf"), "negative_-0.2": -0.2,
             "above_one_1.3": 1.3, "None": None, "string_'0.5'": "0.5", "bool_True": True,
             "list_[0.5]": [0.5]}
    out = {}
    for name, val in kinds.items():
        ok = 0
        for _ in range(1000):
            r = router.route(action=lambda d: d, input_data={"a": 1}, risk_score=val, human_approved=True)
            ok += (r.path is ReasoningPath.SAFE and not r.safe and r.action_result is None
                   and any(v.startswith("R0_RiskInput") for v in r.violations))
        out[name] = {"calls": 1000, "failed_closed": ok}
    for name, sig in {"sigma_nan": float("nan"), "sigma_negative": -0.05}.items():
        ok = 0
        for _ in range(1000):
            r = router.route(action=lambda d: d, input_data={"a": 1}, risk_score=0.1,
                             risk_uncertainty=sig, human_approved=True)
            ok += (r.path is ReasoningPath.SAFE and not r.safe)
        out[name] = {"calls": 1000, "failed_closed": ok}
    return out


def _path(score, t2=0.7):
    return np.where(score < 0.3, 0, np.where(score < t2, 1, 2))


def part_b(rho, strata) -> list:
    rows = []
    r = np.clip(rho, 1e-6, 1 - 1e-6)
    lg = np.log(r / (1 - r))
    for a in (0.6, 0.8, 1.0, 1.25):
        for b in (-1.0, -0.5, 0.0, 0.5):
            obs = 1 / (1 + np.exp(-(a * lg + b)))
            p = _path(obs)
            rows.append({"slope_a": a, "intercept_b": b,
                         "mean_shift": round(float(np.mean(obs - rho)), 4),
                         "down_safe_pct": round(100 * float(np.mean(p[strata == "SAFE"] < 2)), 2),
                         "up_to_safe_from_normal_pct": round(
                             100 * float(np.mean(p[strata == "NORMAL"] == 2)), 2),
                         "safe_share_pct": round(100 * float(np.mean(p == 2)), 2)})
    return rows


def part_c() -> list:
    rows = []
    for safe_share in (0.10, 0.20, 0.30, 0.40):
        n = 10000
        n_safe = int(round(n * safe_share))
        n_fast = int(round((n - n_safe) * 35 / 80))
        n_norm = n - n_safe - n_fast
        rng = random.Random(SEED)
        rho = np.array([rng.uniform(0, .3) for _ in range(n_fast)] + [rng.uniform(.3, .7) for _ in range(n_norm)]
                       + [rng.uniform(.7, 1) for _ in range(n_safe)])
        truth = np.array([0] * n_fast + [1] * n_norm + [2] * n_safe)
        for t2 in (0.70, 0.60):
            appr, miss = [], []
            for rep in range(20):
                obs = np.clip(rho + np.random.default_rng([SEED, rep]).normal(0, 0.10, n), 0, 1)
                p = _path(obs, t2)
                appr.append(1000 * np.mean(p == 2))
                miss.append(1000 * np.mean((truth == 2) & (p < 2)))
            rows.append({"safe_prevalence": safe_share, "theta2": t2,
                         "approvals_per_1000": round(float(np.mean(appr)), 1),
                         "missed_approvals_per_1000": round(float(np.mean(miss)), 1)})
    return rows


def run() -> Dict:
    cohort = lb.build_cohort(random.Random(SEED))
    rho = np.array([c["risk"] for c in cohort])
    strata = np.array([c["stratum"] for c in cohort])
    return {"seed": SEED, "fail_closed": part_a(), "calibration_drift": part_b(rho, strata),
            "prevalence_shift": part_c()}


def main() -> None:
    out = run()
    p = ROOT / "results" / "upstream_reliability.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["fail_closed"]))
    for r in out["calibration_drift"]:
        print(r)
    for r in out["prevalence_shift"]:
        print(r)


if __name__ == "__main__":
    main()
