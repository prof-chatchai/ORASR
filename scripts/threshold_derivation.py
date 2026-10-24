"""Decision-theoretic derivation of the routing thresholds (v1.3.0).

Model. Treat a calibrated score rho as the probability that the action causes
harm of cost H if executed with only the Fast-tier controls. G2+G3 (Normal)
prevent a fraction dN of that harm at an extra 2 gate evaluations of cost c;
the Safe tier (approval + G4) prevents a further fraction dS at an extra gate
plus one approval of cost A. Expected cost per pathway:

    Fast   = c       + rho H (1 - d0)
    Normal = 3c      + rho H (1 - d0 - dN)
    Safe   = 4c + A  + rho H (1 - d0 - dN - dS)

so Normal beats Fast iff rho >= theta1* = 2c / (H dN) and Safe beats Normal
iff rho >= theta2* = (c + A) / (H dS). Writing H = r A (harm-to-approval cost
ratio) gives theta2* ~= 1 / (r dS) for c << A.

With a noisy score rho_hat = rho + e the optimal *observed* threshold is found
numerically by minimising the empirical expected cost over a 0.005 grid. Done
for the seed-42 synthetic cohort and, when present, for the risk distribution
observed on MIMIC-IV demo orders (results/mimic_demo_routing.json; same code
reads results/mimic_full_routing.json when a credentialed run exists).
Output: results/threshold_derivation.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402

SEED = lb.SEED
A_S = 30.0
R_GRID = (1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
DS_GRID = (0.25, 0.5, 0.9)
SIGMAS = (0.0, 0.05, 0.10)
C_GRID_S = (0.01, 1.0, 10.0)
GRID = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def theta2_star(r: float, ds: float, c: float = 0.01, a: float = A_S) -> float:
    return min(1.0, (c + a) / (r * a * ds))


def theta1_star(r: float, dn: float, c: float, a: float = A_S) -> float:
    return min(1.0, 2 * c / (r * a * dn))


def optimal_observed_theta2(rho: np.ndarray, r: float, ds: float, sigma: float, n_rep: int = 10,
                            c: float = 0.01) -> Dict[str, float]:
    h = r * A_S
    gain = rho * h * ds - (c + A_S)            # benefit of Safe over Normal, per action
    best = None
    for t in GRID:
        vals, share, missed = [], [], []
        for rep in range(n_rep if sigma > 0 else 1):
            obs = rho if sigma == 0 else np.clip(
                rho + np.random.default_rng([SEED, rep]).normal(0, sigma, len(rho)), 0, 1)
            safe = obs >= t
            vals.append(-np.mean(np.where(safe, gain, 0.0)))
            share.append(np.mean(safe))
            missed.append(np.mean((~safe) & (gain > 0)))
        cost = float(np.mean(vals))
        if best is None or cost < best["rel_cost_s"] - 1e-12:
            best = {"theta2_obs": float(t), "rel_cost_s": cost, "safe_share_pct": 100 * float(np.mean(share)),
                    "missed_beneficial_approvals_pct": 100 * float(np.mean(missed))}
    return {k: round(v, 4) for k, v in best.items()}


def load_mimic(label: str) -> Optional[np.ndarray]:
    p = ROOT / "results" / f"mimic_{label}_routing.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    vals = d["mapping_v2"]["risk_value_counts"]
    return np.array([float(k) for k, n in vals.items() for _ in range(int(n))])


def sweep(rho: np.ndarray) -> List[Dict]:
    rows = []
    for r in R_GRID:
        for ds in DS_GRID:
            row = {"r_harm_over_approval": r, "dS": ds, "theta2_star_analytic": round(theta2_star(r, ds), 4)}
            for s in SIGMAS:
                row[f"sigma_{s:.2f}"] = optimal_observed_theta2(rho, r, ds, s)
            rows.append(row)
    return rows


def run() -> Dict:
    cohort = lb.build_cohort(random.Random(SEED))
    synth = np.array([c["risk"] for c in cohort])
    out = {
        "seed": SEED, "approval_cost_s": A_S, "gate_cost_s_for_theta2": 0.01,
        "theta1_star": [{"c_s": c, "r": r, "dN": dn, "theta1_star": round(theta1_star(r, dn, c), 6)}
                        for c in C_GRID_S for r in (3.0, 10.0) for dn in (0.25, 0.5)],
        "default_theta2_0.7_implies": [{"dS": ds, "r_for_theta2_0.7": round((0.01 + A_S) / (0.7 * A_S * ds), 3)}
                                       for ds in DS_GRID],
        "synthetic": sweep(synth),
    }
    for label in ("demo", "full"):
        m = load_mimic(label)
        if m is not None:
            out[f"mimic_{label}"] = {"n_actions": int(len(m)), "sweep": sweep(m)}
    return out


def main() -> None:
    out = run()
    p = ROOT / "results" / "threshold_derivation.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for row in out["synthetic"]:
        if row["dS"] == 0.5:
            print(row["r_harm_over_approval"], row["theta2_star_analytic"],
                  {k: v["theta2_obs"] for k, v in row.items() if k.startswith("sigma")})
    print(out["theta1_star"])
    for k in ("mimic_demo", "mimic_full"):
        if k in out:
            print(k, [(r["r_harm_over_approval"], r["dS"], r["sigma_0.10"]) for r in out[k]["sweep"]
                      if r["dS"] == 0.5])


if __name__ == "__main__":
    main()
