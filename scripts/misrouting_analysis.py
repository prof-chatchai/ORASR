"""Misrouting under risk-score noise (seed 42, synthetic).

Pathway selection uses the committed ``ORASRRouter._select_pathway`` (the same
function ``route()`` calls). Noisy scores are NOT clipped: the router's own
out-of-range behaviour applies (rho < 0 -> FAST, rho > 1 -> SAFE).

(a) Down- and up-routing per stratum for sigma in SIGMAS. Each action gets
    N_REP independent Gaussian draws. Down: SAFE->NORMAL/FAST, NORMAL->FAST.
    Up: FAST->NORMAL/SAFE, NORMAL->SAFE. Extra approval burden = up-routed
    actions landing in SAFE per 1000 cohort actions.
(b) SAFE-stratum down-routing probability vs distance d = rho - 0.70 in bins of
    0.02, sigma in {0.05, 0.10}: simulated vs analytic Phi(-d/sigma)
    (scipy.stats.norm), analytic averaged over the members of each bin.
(c) Guard band: route with SAFE threshold theta2' in {0.50..0.70} under sigma
    = 0.10. True-Safe = original rho >= 0.70. Reports the down-route rate of
    true-Safe actions and the SAFE share of all actions (approval burden).

Output: results/misrouting_analysis.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402
from orasr import ORASRRouter  # noqa: E402

SEED = lb.SEED
SIGMAS = [0.02, 0.05, 0.10, 0.15, 0.20]
BIN_SIGMAS = [0.05, 0.10]
GUARD_THETAS = [0.50, 0.55, 0.60, 0.65, 0.70]
N_REP = 20
BIN_W = 0.02
RANK = {"FAST": 0, "NORMAL": 1, "SAFE": 2}


def route_names(router: ORASRRouter, values: np.ndarray) -> np.ndarray:
    # Noisy scores are clipped to [0, 1] as a probability-valued upstream would emit
    # them. (Since v1.3.0 the router fails closed on out-of-range scores; that
    # behaviour is analysed separately in scripts/upstream_reliability.py.)
    vals = np.clip(values, 0.0, 1.0)
    return np.array([RANK[router._select_pathway(float(v), False).name] for v in vals])


def part_a(router, strata: np.ndarray, rho: np.ndarray) -> List[Dict]:
    n_cohort = len(rho)
    out = []
    for i, sigma in enumerate(SIGMAS):
        rng = np.random.default_rng([SEED, 1, i])
        noisy = rho[None, :] + rng.normal(0.0, sigma, size=(N_REP, len(rho)))
        routed = route_names(router, noisy.ravel()).reshape(noisy.shape)
        entry = {"sigma": sigma, "replicates": N_REP, "per_stratum": {}}
        extra_safe = 0
        for s in ("FAST", "NORMAL", "SAFE"):
            m = strata == RANK[s]
            r = routed[:, m]
            total = r.size
            counts = {k: int((r == v).sum()) for k, v in RANK.items()}
            down = int((r < RANK[s]).sum())
            up = int((r > RANK[s]).sum())
            up_to_safe = int((r == 2).sum()) if s != "SAFE" else 0
            extra_safe += up_to_safe
            entry["per_stratum"][s] = {
                "n_actions": int(m.sum()), "n_routings": total, "routed_counts": counts,
                "down_routed_pct": round(100 * down / total, 3),
                "up_routed_pct": round(100 * up / total, 3),
                "up_routed_to_safe_pct": round(100 * up_to_safe / total, 3),
            }
        entry["extra_safe_approvals_per_1000_actions"] = round(
            1000 * extra_safe / (N_REP * n_cohort), 3)
        entry["lost_safe_per_1000_actions"] = round(
            1000 * int((routed[:, strata == 2] < 2).sum()) / (N_REP * n_cohort), 3)
        out.append(entry)
    return out


def part_b(router, safe_rho: np.ndarray) -> List[Dict]:
    theta = router.SAFE_PATH_THRESHOLD
    d = safe_rho - theta
    n_bins = int(round((d.max() + 1e-12) / BIN_W)) + 1
    out = []
    for i, sigma in enumerate(BIN_SIGMAS):
        rng = np.random.default_rng([SEED, 2, i])
        noisy = safe_rho[None, :] + rng.normal(0.0, sigma, size=(N_REP * 10, len(safe_rho)))
        down = (route_names(router, noisy.ravel()).reshape(noisy.shape) < 2)
        bins = []
        for b in range(n_bins):
            lo, hi = b * BIN_W, (b + 1) * BIN_W
            m = (d >= lo) & (d < hi)
            if not m.any():
                continue
            sim = float(down[:, m].mean())
            ana = float(norm.cdf(-d[m] / sigma).mean())
            bins.append({"d_lo": round(lo, 2), "d_hi": round(hi, 2), "n_actions": int(m.sum()),
                         "n_draws": int(down[:, m].size), "simulated": round(sim, 5),
                         "analytic_phi": round(ana, 5), "abs_diff": round(abs(sim - ana), 5)})
        out.append({"sigma": sigma, "draws_per_action": N_REP * 10, "bins": bins,
                    "max_abs_diff": max(x["abs_diff"] for x in bins),
                    "overall_simulated": round(float(down.mean()), 5),
                    "overall_analytic": round(float(norm.cdf(-d / sigma).mean()), 5)})
    return out


def part_c(strata: np.ndarray, rho: np.ndarray) -> List[Dict]:
    sigma = 0.10
    rng = np.random.default_rng([SEED, 3, 0])
    noise = rng.normal(0.0, sigma, size=(N_REP, len(rho)))
    true_safe = rho >= ORASRRouter.SAFE_PATH_THRESHOLD
    out = []
    for theta2 in GUARD_THETAS:
        router = ORASRRouter(enable_audit=False)
        router.SAFE_PATH_THRESHOLD = theta2  # instance override; engine code unchanged
        routed = route_names(router, (rho[None, :] + noise).ravel()).reshape(noise.shape)
        clean = route_names(router, rho)
        out.append({
            "theta2_prime": theta2, "sigma": sigma, "replicates": N_REP,
            "true_safe_down_routed_pct": round(100 * float((routed[:, true_safe] < 2).mean()), 3),
            "safe_share_all_actions_pct": round(100 * float((routed == 2).mean()), 3),
            "safe_share_no_noise_pct": round(100 * float((clean == 2).mean()), 3),
            "non_true_safe_routed_safe_pct": round(
                100 * float((routed[:, ~true_safe] == 2).mean()), 3),
        })
    return out


def run() -> Dict:
    cohort = lb.build_cohort(random.Random(SEED))
    rho = np.array([c["risk"] for c in cohort])
    strata = np.array([RANK[c["stratum"]] for c in cohort])
    router = ORASRRouter(enable_audit=False)
    return {
        "seed": SEED,
        "rng": "numpy.random.default_rng([42, part, sigma_index])",
        "noise_model": "rho_noisy = rho + N(0, sigma^2), not clipped",
        "a_down_up_routing": part_a(router, strata, rho),
        "b_down_routing_vs_distance": part_b(router, rho[strata == 2]),
        "c_guard_band_sweep": part_c(strata, rho),
    }


def main() -> None:
    out = run()
    path = ROOT / "results" / "misrouting_analysis.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for e in out["a_down_up_routing"]:
        ps = e["per_stratum"]
        print(f"sigma={e['sigma']}: SAFE down {ps['SAFE']['down_routed_pct']}% | NORMAL down "
              f"{ps['NORMAL']['down_routed_pct']}% up->SAFE {ps['NORMAL']['up_routed_to_safe_pct']}%"
              f" | FAST up {ps['FAST']['up_routed_pct']}% | extra approvals/1000 "
              f"{e['extra_safe_approvals_per_1000_actions']}")
    for e in out["b_down_routing_vs_distance"]:
        print(f"(b) sigma={e['sigma']} max|sim-analytic|={e['max_abs_diff']} overall "
              f"{e['overall_simulated']} vs {e['overall_analytic']}")
    for e in out["c_guard_band_sweep"]:
        print("(c)", e)
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
