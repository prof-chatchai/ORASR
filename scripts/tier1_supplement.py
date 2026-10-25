"""Supplementary analyses for the second review of the manuscript (v1.3.1).

Adds uncertainty, harmonised error models, analytic cross-checks and
clinically refined sensitivity analyses to the v1.3.0 results WITHOUT changing
any v1.3.0 result file. Deterministic (seed 42). Writes results/tier1_supplement.json.

  A  mitigation grid with replicate SD, 95% CI and paired differences
     (same 20 replicate streams as scripts/structured_misrouting.py; the means
     reproduce results/mitigation_comparison.json, which is checked)
  B  E3 and E5 re-run with the same background noise (sigma = 0.10) as E1/E4
  C  E7 attack-size sweep against every mitigation (guard band vs attack size)
  D  equivalence of the vectorised router of structured_misrouting.py with
     ORASRRouter.route (pathway agreement on every action of every model)
  E  misrouting: cohort-conditional exact expectation and Monte Carlo SE
  F  MIMIC-IV demo: patient-level cluster-bootstrap CIs for pathway shares and
     approvals; Wilson CIs for fault detection on real orders; approvals per
     patient-day; structural escape of tenfold doses under absolute ceilings;
     potential duplicates by ingredient; a refined, route-aware mapping v2r
     (heparin/saline flushes excluded, route-dependent ISMP classes Safe only
     when given IV, oral opioids incl. tramadol, respiratory and ADT orders on
     Normal); rescue-trigger counts per class
  G  threshold derivation extended: harm-to-approval ratios up to 10^4,
     false-block cost, and the condition for three tiers to be optimal
  H  approval delay: closed form versus the Monte Carlo of assurance_mechanisms.py
  I  latency: analytic saturation throughput versus the measured throughput
  J  Wilson CI for the ED acuity disagreement

Usage: python -I scripts/tier1_supplement.py --mimic-path D:/AVS/Datasets/mimic-iv-demo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import structured_misrouting as sm  # noqa: E402
from orasr import ORASRRouter  # noqa: E402

logging.disable(logging.CRITICAL)
SEED = 42
RES = ROOT / "results"
T975_19 = 2.093  # t quantile, 19 degrees of freedom


def _ci(v) -> Dict[str, float]:
    v = np.asarray(v, dtype=float)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    h = T975_19 * sd / math.sqrt(len(v))
    return {"mean": round(m, 3), "sd": round(sd, 3), "ci95": [round(m - h, 3), round(m + h, 3)]}


def wilson(k: int, n: int, z: float = 1.959964) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------- A, B, C
def observe_harmonised(model, rho, subgroup, enc, rng):
    """E3/E5 with background noise sigma = 0.10 (as E1/E4)."""
    n = len(rho)
    sig_hat = rng.uniform(0.02, 0.18, n)
    corrupted = np.zeros(n, dtype=bool)
    if model == "E3_subgroup_bias_s10":
        obs = rho + np.where(subgroup, -0.15, 0.0) + rng.normal(0, 0.10, n)
    elif model == "E5_common_mode_s10":
        bad = rng.random(enc.max() + 1) < 0.10
        corrupted = bad[enc]
        obs = rho + np.where(corrupted, -0.20, 0.0) + rng.normal(0, 0.10, n)
    else:
        raise ValueError(model)
    return np.clip(obs, 0, 1), sig_hat, corrupted


def grid_with_ci() -> Dict:
    rho, strata, subgroup, enc, hi_alert = sm.setup()
    ed = sm.load_ed()
    true_safe, true_ns = strata == "SAFE", strata != "FAST"
    n = len(rho)
    models = [m for m in sm.ERROR_MODELS if m != "E8_empirical_ED" or ed]
    harmonised = ("E3_subgroup_bias_s10", "E5_common_mode_s10")
    per = {}
    for model in list(models) + list(harmonised):
        per[model] = {mit: defaultdict(list) for mit in sm.MITIGATIONS}
        for rep in range(sm.N_REP):
            if model in harmonised:
                idx = sm.ERROR_MODELS.index(model.replace("_s10", ""))
                rng = np.random.default_rng([SEED, rep, idx, 10])
                obs, sig_hat, corrupted = observe_harmonised(model, rho, subgroup, enc, rng)
            else:
                rng = np.random.default_rng([SEED, rep, sm.ERROR_MODELS.index(model)])
                obs, sig_hat, corrupted = sm.observe(model, rho, strata, subgroup, enc, rng, ed)
            frng = np.random.default_rng([SEED, rep, 99])
            fault = true_ns & (frng.random(n) < 0.10)
            for mit in sm.MITIGATIONS:
                path = sm.route(mit, obs, sig_hat, hi_alert, corrupted)
                blind = corrupted & (mit != "M5_completeness_check_plus_floor")
                missed = fault & ((path == 0) | blind)
                d = per[model][mit]
                d["down_safe_pct"].append(100 * np.mean(path[true_safe] < 2))
                d["down_safe_subgroup_pct"].append(100 * np.mean(path[true_safe & subgroup] < 2))
                d["down_safe_nonsubgroup_pct"].append(100 * np.mean(path[true_safe & ~subgroup] < 2))
                d["safe_share_pct"].append(100 * np.mean(path == 2))
                d["overdose_missed_pct"].append(100 * missed.sum() / max(1, fault.sum()))
                d["gates_per_call"].append(float(np.array([1, 3, 4])[path].mean()))
    summary = {m: {mit: {k: _ci(v) for k, v in d.items()} for mit, d in per[m].items()} for m in per}
    # consistency with the published grid
    pub = json.loads((RES / "mitigation_comparison.json").read_text(encoding="utf-8"))["grid"]
    max_dev = max(abs(summary[m][mit][k]["mean"] - pub[m][mit][k])
                  for m in models for mit in sm.MITIGATIONS
                  for k in ("down_safe_pct", "safe_share_pct", "overdose_missed_pct", "gates_per_call"))
    pairs = (("M1_guard_band_0.60", "M2_uncertainty_k1"), ("M1_guard_band_0.60", "M0_none"),
             ("M2_uncertainty_k1", "M0_none"), ("M4_guard_plus_floor", "M3_type_floor"),
             ("M5_completeness_check_plus_floor", "M4_guard_plus_floor"))
    paired = {}
    for m in per:
        paired[m] = {}
        for a, b in pairs:
            for k in ("down_safe_pct", "safe_share_pct", "overdose_missed_pct"):
                diff = np.array(per[m][a][k]) - np.array(per[m][b][k])
                paired[m][f"{a} - {b}: {k}"] = _ci(diff)
    return {"n_replicates": sm.N_REP, "ci_method": "mean +/- t(0.975, 19) * SD / sqrt(20) over replicates",
            "summary": summary, "paired_differences": paired,
            "max_abs_deviation_from_mitigation_comparison_json": round(max_dev, 4),
            "harmonised_models": {"E3_subgroup_bias_s10": "subgroup bias 0.15 + N(0, 0.10^2) for all",
                                  "E5_common_mode_s10": "10% corrupted encounters shift -0.20 + N(0, 0.10^2) for all"},
            "noise_by_model": {"E1": "N(0,0.10^2)", "E2": "N(0,s^2), s~U(0.02,0.18)",
                               "E3": "subgroup -0.15 + N(0,0.05^2)", "E4": "total SD 0.10, ICC 0.81",
                               "E5": "corrupted -0.20 (no noise); others N(0,0.05^2)",
                               "E6": "deterministic drift", "E7": "deterministic -0.10", "E8": "empirical resampling"}}


def attack_sweep() -> Dict:
    rho, strata, subgroup, enc, hi_alert = sm.setup()
    true_safe = strata == "SAFE"
    n = len(rho)
    sig = np.random.default_rng([SEED, 7]).uniform(0.02, 0.18, n)
    out = []
    for delta in (0.05, 0.10, 0.15, 0.20, 0.30):
        obs = np.clip(rho - delta, 0, 1)
        row = {"delta": delta}
        for mit in sm.MITIGATIONS:
            path = sm.route(mit, obs, sig, hi_alert, np.zeros(n, bool))
            row[mit] = {"down_safe_pct": round(100 * float(np.mean(path[true_safe] < 2)), 2),
                        "safe_share_pct": round(100 * float(np.mean(path == 2)), 2)}
        out.append(row)
    return {"rows": out, "note": "non-adaptive subtraction; uninformative sigma_hat ~ U(0.02, 0.18)"}


# ---------------------------------------------------------------- D
def router_equivalence() -> Dict:
    rho, strata, subgroup, enc, hi_alert = sm.setup()
    ed = sm.load_ed()
    n = len(rho)
    models = [m for m in sm.ERROR_MODELS if m != "E8_empirical_ED" or ed]
    routers = {
        "M0_none": ORASRRouter(enable_audit=False),
        "M1_guard_band_0.60": ORASRRouter(enable_audit=False),
        "M2_uncertainty_k1": ORASRRouter(enable_audit=False, uncertainty_k=1.0),
        "M3_type_floor": ORASRRouter(enable_audit=False, high_alert_types={"high_alert"}),
    }
    routers["M1_guard_band_0.60"].SAFE_PATH_THRESHOLD = 0.60
    code = {"FAST": 0, "NORMAL": 1, "SAFE": 2}
    checked = mism = 0
    for model in models:
        rng = np.random.default_rng([SEED, 0, sm.ERROR_MODELS.index(model)])
        obs, sig_hat, corrupted = sm.observe(model, rho, strata, subgroup, enc, rng, ed)
        for mit, r in routers.items():
            vec = sm.route(mit, obs, sig_hat, hi_alert, corrupted)
            for i in range(n):
                kw = {"human_approved": True}
                if mit == "M2_uncertainty_k1":
                    kw["risk_uncertainty"] = float(sig_hat[i])
                if mit == "M3_type_floor":
                    kw["action_type"] = "high_alert" if hi_alert[i] else "other"
                res = r.route(lambda d: d, {"i": i}, float(obs[i]), **kw)
                checked += 1
                mism += code[res.path.name] != int(vec[i])
    return {"calls_checked": checked, "pathway_mismatches": mism,
            "scope": "replicate 0 of every error model x mitigations M0-M3 through ORASRRouter.route"}


# ---------------------------------------------------------------- E
def misrouting_exact() -> Dict:
    rho, strata, *_ = sm.setup()
    safe = rho[strata == "SAFE"]
    rows = []
    for t2 in (0.50, 0.55, 0.60, 0.65, 0.70):
        exact = float(np.mean([phi((t2 - r) / 0.10) for r in safe]))
        se = math.sqrt(exact * (1 - exact) / (len(safe) * 20))
        rows.append({"theta2": t2, "cohort_conditional_pct": round(100 * exact, 2),
                     "mc_se_pp_20_replicates": round(100 * se, 2)})
    return {"sigma": 0.10, "n_safe": int(len(safe)), "rows": rows,
            "note": "expectation of Phi((theta2 - rho)/sigma) over the seed-42 Safe stratum; the "
                    "closed form of Eq. (guard) assumes rho uniform on [0.7, 1]"}


# ---------------------------------------------------------------- F
IV_ROUTES = ("iv", "iv drip", "iv bolus", "ivpb", "iv lock", "intravenous", "ivs", "iv continuous")
ROUTE_DEPENDENT = ("potassium chloride", "potassium phosphate", "magnesium sulfate", "lorazepam",
                   "midazolam", "digoxin", "amiodarone")


def mimic_block(mimic: Path) -> Dict:
    import mimic_routing as mr
    from clinical_config import dose_limit, duplicate_ok, ingredient_of, parse_dose
    rx = list(mr.read_csv(mimic / "hosp" / "prescriptions.csv.gz"))
    poe = list(mr.read_csv(mimic / "hosp" / "poe.csv.gz"))
    adm = list(mr.read_csv(mimic / "hosp" / "admissions.csv.gz"))
    poe_nonmed = [r for r in poe if r.get("order_type", "") != "Medications"]
    router = ORASRRouter(enable_audit=False)

    def path_of(rho):
        return router._select_pathway(rho, False).name

    def v2r_med(r):
        t = r.get("drug", "").lower()
        route = r.get("route", "").strip().lower()
        if mr._has(t, ("flush",)):
            return 0.15
        if mr._has(t, ROUTE_DEPENDENT):
            return 0.85 if route in IV_ROUTES else 0.50
        if mr._has(t, mr.HIGH_ALERT) or mr._has(t, ("tramadol",)):
            return 0.85
        if mr._has(t, mr.LOW_RISK):
            return 0.15
        return 0.50

    def v2r_poe(r):
        t = r.get("order_type", "").lower()
        if mr._has(t, mr.POE_HIGH):
            return 0.85
        if mr._has(t, ("adt orders", "respiratory")):
            return 0.50
        if mr._has(t, mr.POE_LOW):
            return 0.15
        return 0.50

    recs = {"v2": [], "v2r": []}
    for r in rx:
        s = r.get("subject_id", "")
        recs["v2"].append((s, path_of(mr.score_v2_med(r.get("drug", "")))))
        recs["v2r"].append((s, path_of(v2r_med(r))))
    for r in poe_nonmed:
        s = r.get("subject_id", "")
        recs["v2"].append((s, path_of(mr.score_v2_poe(r.get("order_type", ""), ""))))
        recs["v2r"].append((s, path_of(v2r_poe(r))))

    # patient-days
    pdays = 0.0
    for a in adm:
        t0, t1 = mr.ts(a.get("admittime", "")), mr.ts(a.get("dischtime", ""))
        if t0 and t1 and t1 > t0:
            pdays += (t1 - t0).total_seconds() / 86400

    brng = np.random.default_rng(SEED)

    def summarise(rows):
        subjects = sorted({s for s, _ in rows})
        idx = {s: i for i, s in enumerate(subjects)}
        cnt = np.zeros((len(subjects), 3))
        for s, p in rows:
            cnt[idx[s], ("FAST", "NORMAL", "SAFE").index(p)] += 1
        tot = cnt.sum(0)
        point = 100 * tot / tot.sum()
        boots = []
        for _ in range(2000):
            pick = brng.integers(0, len(subjects), len(subjects))
            c = cnt[pick].sum(0)
            boots.append(np.concatenate([100 * c / c.sum(), [(c[0] + 3 * c[1] + 4 * c[2]) / c.sum()]]))
        b = np.array(boots)
        lo, hi = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
        g = (tot[0] + 3 * tot[1] + 4 * tot[2]) / tot.sum()
        names = ("fast_pct", "normal_pct", "safe_pct", "gates_per_call")
        vals = list(point) + [g]
        out = {k: {"value": round(float(v), 2), "ci95": [round(float(l), 2), round(float(h), 2)]}
               for k, v, l, h in zip(names, vals, lo, hi)}
        out["approvals_per_1000"] = {"value": round(10 * float(point[2]), 1),
                                     "ci95": [round(10 * float(lo[2]), 1), round(10 * float(hi[2]), 1)]}
        out["n_actions"] = int(tot.sum())
        out["n_patients"] = len(subjects)
        out["approvals_per_patient_day"] = round(float(tot[2]) / pdays, 1) if pdays else None
        return out

    shares = {k: summarise(v) for k, v in recs.items()}
    moved = Counter()
    for r in rx:
        a, b = path_of(mr.score_v2_med(r.get("drug", ""))), path_of(v2r_med(r))
        if a != b:
            moved[(r.get("drug", "").strip().lower(), r.get("route", "").strip().lower(), a, b)] += 1
    poe_moved = Counter()
    for r in poe_nonmed:
        a, b = path_of(mr.score_v2_poe(r.get("order_type", ""), "")), path_of(v2r_poe(r))
        if a != b:
            poe_moved[(r.get("order_type", ""), a, b)] += 1

    # structural tenfold escape and potential duplicates (mapping v2 pathways)
    esc = defaultdict(Counter)
    cov = Counter()
    dup = defaultdict(Counter)
    n_path = Counter()
    by_adm = defaultdict(list)
    for r in rx:
        st = mr.ts(r.get("starttime", ""))
        if st is None:
            continue
        sp = mr.ts(r.get("stoptime", "")) or st + timedelta(hours=24)
        by_adm[r.get("hadm_id", "")].append((r, st, sp))
    for orders in by_adm.values():
        orders.sort(key=lambda o: o[1])
        for i, (r, st, sp) in enumerate(orders):
            drug = r.get("drug", "")
            p = path_of(mr.score_v2_med(drug))
            n_path[p] += 1
            lim, d0 = dose_limit(drug, r.get("dose_unit_rx", "")), parse_dose(r.get("dose_val_rx", ""))
            if lim is not None and d0 is not None and d0 > 0:
                cov[p] += 1
                if 10 * d0 <= lim:
                    esc[p][ingredient_of(drug)] += 1
            active = [q.get("drug", "") for j, (q, s2, e2) in enumerate(orders) if j != i and s2 <= st < e2]
            if not duplicate_ok({"drug": drug, "active_meds": active}):
                dup[p][ingredient_of(drug)] += 1
    escape = {p: {"dose_covered": cov[p], "tenfold_within_ceiling": sum(esc[p].values()),
                  "pct": round(100 * sum(esc[p].values()) / cov[p], 1) if cov[p] else None,
                  "top": [{"ingredient": k, "n": v} for k, v in esc[p].most_common(5)]}
              for p in ("FAST", "NORMAL", "SAFE")}
    dups = {p: {"n": sum(dup[p].values()), "pct": round(100 * sum(dup[p].values()) / n_path[p], 1),
                "top": [{"ingredient": k, "n": v} for k, v in dup[p].most_common(6)]}
            for p in ("FAST", "NORMAL", "SAFE")}

    pub = json.loads((RES / "mimic_demo_routing.json").read_text(encoding="utf-8"))
    inj = pub["clinical_gates_on_real_orders"]["fault_injection_on_real_orders"]
    wil = {p: {f: {"detected": c["detected"], "injected": c["injected"],
                   "detected_pct": round(100 * c["detected"] / c["injected"], 1) if c["injected"] else None,
                   "wilson95": wilson(c["detected"], c["injected"])}
               for f, c in d.items()} for p, d in inj.items()}
    trig = pub["rescue_trigger_analysis"]
    return {"cluster_bootstrap": {"method": "2,000 resamples of patients (subject_id), percentile 95% CI",
                                  "patient_days": round(pdays, 1), **shares},
            "mapping_v2r_definition": ("v2 plus: heparin/saline flushes Fast; potassium chloride/phosphate, "
                                       "magnesium sulfate, lorazepam, midazolam, digoxin, amiodarone Safe only "
                                       "by an IV route (ISMP lists these classes for IV/concentrated use), else "
                                       "Normal; tramadol added (oral opioids are ISMP high-alert); POE "
                                       "respiratory and ADT orders moved from Fast to Normal"),
            "v2_to_v2r_top_moved_prescriptions": [{"drug": k[0], "route": k[1], "from": k[2], "to": k[3], "n": v}
                                                  for k, v in moved.most_common(10)],
            "v2_to_v2r_moved_prescriptions_total": int(sum(moved.values())),
            "v2_to_v2r_moved_poe": [{"order_type": k[0], "from": k[1], "to": k[2], "n": v}
                                    for k, v in poe_moved.most_common()],
            "tenfold_dose_structural_escape": escape,
            "potential_duplicates_by_ingredient": dups,
            "fault_injection_wilson": wil,
            "rescue_triggers": {"per_trigger": trig["per_trigger"], "v1_triggered_on_safe_pct": trig["triggered_on_safe_pct_v1"],
                                "v2_triggered_on_safe_pct": trig["triggered_on_safe_pct_v2"],
                                "implicated_index_not_safe_v2": trig["total_index_orders_v2"].get("NORMAL", 0)}}


# ---------------------------------------------------------------- G, H, I, J
def threshold_extended() -> Dict:
    A = 30.0
    rows = []
    for r in (3, 10, 30, 100, 1000, 10000):
        for eta in (0.25, 0.5, 0.9):
            rows.append({"r": r, "eta_S": eta, "theta2_star": round(min(1.0, (0.01 + A) / (r * A * eta)), 5)})
    fb = []
    # theta1* with a false-block cost: Normal beats Fast iff rho H eta_N >= 2c + q_N B
    for q_n in (0.0, 0.088):
        for b_over_a in (1.0, 10.0):
            for r in (10, 100, 1000):
                for c in (0.01, 10.0):
                    t1 = min(1.0, (2 * c + q_n * b_over_a * A) / (r * A * 0.5))
                    t2 = min(1.0, (c + A) / (r * A * 0.5))
                    fb.append({"q_N": q_n, "B_over_A": b_over_a, "r": r, "c_s": c, "eta": 0.5,
                               "theta1_star": round(t1, 5), "theta2_star": round(t2, 5),
                               "three_tiers_optimal": t1 < t2})
    return {"theta2_star_vs_r": rows, "false_block_extension": fb,
            "model": ("Fast vs Normal: rho*H*eta_N >= 2c + q_N*B (q_N = false-block probability of G2/G3 on "
                      "a harmless action, B = cost of a false block); Normal vs Safe: rho*H*eta_S >= c + A + q_S*B "
                      "(q_S = 0 here: approval and G4 add no automated block). Three tiers are optimal only if "
                      "theta1* < theta2*; otherwise the Normal tier is never cost-optimal."),
            "q_N_source": "8.8% firing on unmodified Normal-pathway prescriptions (Table realchecks) as an upper bound"}


def delay_closed_form() -> Dict:
    d = json.loads((RES / "assurance_mechanisms.json").read_text(encoding="utf-8"))["approval_delay"]["rows"]
    out = []
    for row in d:
        mu, D, W = math.log(row["median_response_s"]), row["deadline_s"], row["escalate_after_s"]
        p1 = 1 - phi((math.log(D) - mu) / 1.0)
        if W is None:
            pf, pe = p1, None
        else:
            p2 = 1 - phi((math.log(D - W) - mu) / 1.0) if D > W else 1.0
            pf, pe = p1 * p2, 1 - phi((math.log(W) - mu) / 1.0)
        out.append({"median_s": row["median_response_s"], "escalate_after_s": W,
                    "fallback_closed_form_pct": round(100 * pf, 2), "fallback_mc_pct": row["default_safe_fallback_pct"],
                    "escalated_closed_form_pct": None if pe is None else round(100 * pe, 2),
                    "escalated_mc_pct": row["escalations_pct"]})
    return {"formula": "P(fallback) = P(T1 > D) * P(T2 > D - W); P(escalate) = P(T1 > W); T ~ lognormal(ln m, 1)",
            "rows": out, "max_abs_diff_pp": round(max(abs(r["fallback_closed_form_pct"] - r["fallback_mc_pct"]) for r in out), 2)}


def latency_bound() -> Dict:
    d = json.loads((RES / "latency_realistic.json").read_text(encoding="utf-8"))
    m = d["service_model"]["constraint_validation"]
    es3 = m["median_s"] * math.exp(m["log_sd"] ** 2 / 2)
    share = 0.65  # Normal + Safe share of the cohort mix
    cap = 8
    meas = {k: next(x for x in d["results"]["g3_capacity_8"][k] if x["concurrency"] == 64)["throughput_rps"]
            for k in d["results"]["g3_capacity_8"]}
    return {"E_S3_ms": round(1000 * es3, 3), "realised_mean_S3_ms": d["realised_mean_service_ms"]["constraint_validation"],
            "bound_flat4_rps": round(cap / es3, 0), "bound_orasr_rps": round(cap / (share * es3), 0),
            "bound_ratio": round(1 / share, 3), "measured_at_64": meas,
            "measured_ratio": round(meas.get("orasr", 0) / meas.get("flat4", 1), 3) if "flat4" in meas else None,
            "note": "G3 with 8 servers is the bottleneck: throughput <= 8 / E[S3] (Flat-4) and 8 / (0.65 E[S3]) (ORASR)"}


def ed_ci() -> Dict:
    return {"esi12_below_safe": 100, "esi12_n": 103, "pct": round(100 * 100 / 103, 1), "wilson95": wilson(100, 103)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-path", default=os.environ.get("MIMIC_IV_DEMO_PATH", "D:/AVS/Datasets/mimic-iv-demo"))
    ap.add_argument("--skip-equivalence", action="store_true")
    a = ap.parse_args()
    out = {"seed": SEED, "version": "1.3.1",
           "A_B_mitigation_grid_ci": grid_with_ci(), "C_attack_sweep": attack_sweep(),
           "E_misrouting_exact": misrouting_exact(), "G_threshold_extended": threshold_extended(),
           "H_delay_closed_form": delay_closed_form(), "I_latency_bound": latency_bound(), "J_ed_ci": ed_ci()}
    if not a.skip_equivalence:
        out["D_router_equivalence"] = router_equivalence()
    mp = Path(a.mimic_path)
    if (mp / "hosp" / "prescriptions.csv.gz").exists():
        out["F_mimic_demo"] = mimic_block(mp)
    (RES / "tier1_supplement.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/tier1_supplement.json")


if __name__ == "__main__":
    main()
