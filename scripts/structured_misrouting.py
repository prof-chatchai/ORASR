"""Structured error models x mitigations for risk-score misrouting (v1.3.0).

Deterministic (seed 42, 20 replicates). True scores come from the seed-42
synthetic cohort (scripts/latency_benchmark.build_cohort). Each action also
carries: a subgroup flag (30%), an encounter id (encounters of 5 actions), a
high-alert action-type flag (50% of Safe-stratum, 10% of Normal-stratum
actions; illustrative), and an upstream-reported uncertainty sigma_hat.

Error models (observed rho_hat = clip(rho + e, 0, 1) unless stated)
  E1 gaussian       e ~ N(0, 0.10^2)
  E2 heteroscedastic e ~ N(0, s_i^2), s_i ~ U(0.02, 0.18), sigma_hat = s_i (informative)
  E3 subgroup bias  subgroup: e ~ N(-0.15, 0.05^2); others N(0, 0.05^2)
  E4 correlated     e = u_enc + v, u ~ N(0, 0.09^2) shared by the encounter,
                    v ~ N(0, 0.0436^2): total SD 0.10, intra-encounter corr 0.81
  E5 common-mode    10% of encounters have a corrupted shared input feature:
                    e = -0.20 for all their actions AND the dose-range check in
                    G2 is blind for them (it reads the same feature); others N(0,0.05^2)
  E6 calibration    rho_hat = expit(0.8 * logit(rho) - 0.5) (systematic under-estimation)
  E7 attack         e = -0.10 for every action
  E8 empirical ED   rho_hat drawn from the rule-score distribution observed on the
                    MIMIC-IV-ED demo for the same reference band
                    (results/ed_acuity_disagreement.json)
Except under E2, sigma_hat is uninformative: s ~ U(0.02, 0.18) drawn independently.

Mitigations
  M0 none            theta2 = 0.70
  M1 guard band      theta2 = 0.60
  M2 uncertainty     route on rho_hat + 1.0 * sigma_hat
  M3 type floor      high-alert action types always Safe
  M4 guard + floor   M1 and M3
  M5 independent completeness check: a G1 check that the shared feature is
                     present; when it is missing the action is routed Safe and
                     held (fail-closed) until the feature is restored; + M3

Metrics (mean over replicates)
  down_safe_pct      truly-Safe (rho >= 0.7) actions routed to a lighter pathway
  down_safe_hi_alert_pct  same, restricted to high-alert types
  skip_g23_pct       truly Normal-or-Safe actions routed Fast (G2, G3 skipped)
  safe_share_pct     share of all actions on Safe (approval load)
  overdose_missed_pct  overdose faults (injected on 10% of truly Normal-or-Safe
                     actions) that pass undetected: routed Fast, or dose check blind
Output: results/structured_misrouting.json, results/mitigation_comparison.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402

SEED = lb.SEED
N_REP = 20
ENC_SIZE = 5
T1, T2 = 0.3, 0.7
ERROR_MODELS = ("E0_none", "E1_gaussian", "E2_heteroscedastic", "E3_subgroup_bias",
                "E4_correlated_encounter", "E5_common_mode", "E6_calibration_drift",
                "E7_attack", "E8_empirical_ED")
MITIGATIONS = ("M0_none", "M1_guard_band_0.60", "M2_uncertainty_k1", "M3_type_floor",
               "M4_guard_plus_floor", "M5_completeness_check_plus_floor")


def load_ed() -> Dict[str, np.ndarray]:
    p = ROOT / "results" / "ed_acuity_disagreement.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    conf = d["confusion_reference_band_vs_rule_band"]
    rho_of = {"FAST": 0.15, "NORMAL": 0.50, "SAFE": 0.85}
    pools = {}
    for ref, row in conf.items():
        pools[ref] = np.array([rho_of[b] for b, n in row.items() for _ in range(n)])
    return pools


def setup():
    cohort = lb.build_cohort(random.Random(SEED))
    rho = np.array([c["risk"] for c in cohort])
    strata = np.array([c["stratum"] for c in cohort])
    rng = np.random.default_rng(SEED)
    n = len(rho)
    subgroup = rng.random(n) < 0.30
    enc = np.arange(n) // ENC_SIZE
    p_hi = np.where(strata == "SAFE", 0.5, np.where(strata == "NORMAL", 0.10, 0.0))
    hi_alert = rng.random(n) < p_hi
    return rho, strata, subgroup, enc, hi_alert


def observe(model, rho, strata, subgroup, enc, rng, ed_pools):
    n = len(rho)
    sig_hat = rng.uniform(0.02, 0.18, n)
    corrupted = np.zeros(n, dtype=bool)
    if model == "E0_none":
        obs = rho.copy()
    elif model == "E1_gaussian":
        obs = rho + rng.normal(0, 0.10, n)
    elif model == "E2_heteroscedastic":
        s = rng.uniform(0.02, 0.18, n)
        obs = rho + rng.normal(0, 1, n) * s
        sig_hat = s
    elif model == "E3_subgroup_bias":
        obs = rho + np.where(subgroup, -0.15, 0.0) + rng.normal(0, 0.05, n)
    elif model == "E4_correlated_encounter":
        u = rng.normal(0, 0.09, enc.max() + 1)
        obs = rho + u[enc] + rng.normal(0, 0.0436, n)
    elif model == "E5_common_mode":
        bad_enc = rng.random(enc.max() + 1) < 0.10
        corrupted = bad_enc[enc]
        obs = rho + np.where(corrupted, -0.20, rng.normal(0, 0.05, n))
    elif model == "E6_calibration_drift":
        r = np.clip(rho, 1e-6, 1 - 1e-6)
        obs = 1 / (1 + np.exp(-(0.8 * np.log(r / (1 - r)) - 0.5)))
    elif model == "E7_attack":
        obs = rho - 0.10
    elif model == "E8_empirical_ED":
        obs = np.empty(n)
        for band in ("FAST", "NORMAL", "SAFE"):
            m = strata == band
            obs[m] = rng.choice(ed_pools[band], m.sum())
    else:
        raise ValueError(model)
    return np.clip(obs, 0.0, 1.0), sig_hat, corrupted


def route(mit, obs, sig_hat, hi_alert, corrupted):
    t2 = 0.60 if mit in ("M1_guard_band_0.60", "M4_guard_plus_floor") else T2
    score = obs + sig_hat if mit == "M2_uncertainty_k1" else obs
    path = np.where(score < T1, 0, np.where(score < t2, 1, 2))
    if mit in ("M3_type_floor", "M4_guard_plus_floor", "M5_completeness_check_plus_floor"):
        path = np.where(hi_alert, 2, path)
    if mit == "M5_completeness_check_plus_floor":
        path = np.where(corrupted, 2, path)
    return path


def run() -> Dict:
    rho, strata, subgroup, enc, hi_alert = setup()
    ed_pools = load_ed()
    models = [m for m in ERROR_MODELS if m != "E8_empirical_ED" or ed_pools]
    true_safe = strata == "SAFE"
    true_ns = strata != "FAST"
    grid = {}
    for model in models:
        grid[model] = {}
        for mit in MITIGATIONS:
            acc = {k: [] for k in ("down_safe_pct", "down_safe_hi_alert_pct", "skip_g23_pct",
                                   "safe_share_pct", "overdose_missed_pct", "down_safe_subgroup_pct")}
            for rep in range(N_REP):
                rng = np.random.default_rng([SEED, rep, ERROR_MODELS.index(model)])
                obs, sig_hat, corrupted = observe(model, rho, strata, subgroup, enc, rng, ed_pools)
                path = route(mit, obs, sig_hat, hi_alert, corrupted)
                frng = np.random.default_rng([SEED, rep, 99])
                fault = true_ns & (frng.random(len(rho)) < 0.10)
                blind = corrupted & (mit != "M5_completeness_check_plus_floor")
                missed = fault & ((path == 0) | blind)
                acc["down_safe_pct"].append(100 * np.mean(path[true_safe] < 2))
                acc["down_safe_hi_alert_pct"].append(100 * np.mean(path[true_safe & hi_alert] < 2))
                acc["down_safe_subgroup_pct"].append(100 * np.mean(path[true_safe & subgroup] < 2))
                acc["skip_g23_pct"].append(100 * np.mean(path[true_ns] == 0))
                acc["safe_share_pct"].append(100 * np.mean(path == 2))
                acc["overdose_missed_pct"].append(100 * missed.sum() / max(1, fault.sum()))
            grid[model][mit] = {k: round(float(np.mean(v)), 2) for k, v in acc.items()}
    return {"seed": SEED, "n_replicates": N_REP, "n_actions": int(len(rho)),
            "high_alert_share": {"SAFE": round(float(hi_alert[true_safe].mean()), 4),
                                 "NORMAL": round(float(hi_alert[strata == "NORMAL"].mean()), 4)},
            "error_models": models, "mitigations": list(MITIGATIONS), "grid": grid,
            "ed_pools_n": {k: int(len(v)) for k, v in ed_pools.items()}}


def main() -> None:
    out = run()
    res = ROOT / "results"
    (res / "mitigation_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    structured = {k: v for k, v in out.items() if k != "grid"}
    structured["no_mitigation"] = {m: out["grid"][m]["M0_none"] for m in out["error_models"]}
    (res / "structured_misrouting.json").write_text(json.dumps(structured, indent=2) + "\n",
                                                    encoding="utf-8")
    for m in out["error_models"]:
        print(m)
        for mit in out["mitigations"]:
            g = out["grid"][m][mit]
            print(f"   {mit:34s} down_safe {g['down_safe_pct']:6.2f} hi {g['down_safe_hi_alert_pct']:6.2f}"
                  f" skip23 {g['skip_g23_pct']:6.2f} safe_share {g['safe_share_pct']:6.2f}"
                  f" od_missed {g['overdose_missed_pct']:6.2f}")


if __name__ == "__main__":
    main()
