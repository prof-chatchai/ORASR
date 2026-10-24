"""Deterministic reproducibility driver for ORASR.

Runs only what the committed source code supports, with a fixed seed, and
writes machine-readable artifacts to ``results/``.

What this driver DOES reproduce (deterministic, portable):
  1. Structural properties of the routing engine verified at runtime:
       - pathway-selection thresholds (0.3 / 0.7);
       - monotonic gate inclusion (FAST=1 gate, NORMAL=3, SAFE=4);
       - non-bypass behaviour (a failed gate marks the routing unsafe and the
         action is not executed).
  2. A synthetic routing cohort (3,500 Fast / 4,500 Normal / 2,000 Safe) routed
     through the real ``ORASRRouter``, producing the pathway distribution and an
     honestly measured gate pass rate.
  3. The flat-monitor gate-work baseline (scripts/flat_baseline.py): expected
     gate-evaluations per call (ORASR vs flat), the gate-work reduction %, and
     the work-normalised throughput ratio -- all host-INDEPENDENT integers.
  4. The progressive gate-removal ablation with deterministic failure injection
     (scripts/ablation.py): coverage loss per configuration -> results/ablation.csv.

What this driver does NOT fabricate:
  - Pathway LATENCIES are measured on the current host and written with an
    explicit ``environment`` block. They are wall-clock timings of THIS machine
    and are not portable; no latency literals are copied here (the portable
    gate-work reduction is reproduced instead).
  - The MIMIC-IV retrospective distribution requires credentialed PhysioNet data.
    scripts/extract_mimic.py is runnable against a user-supplied copy but ships
    no data and hardcodes none of the 58.1/31.5/10.4% numbers. Remaining gaps
    are recorded in ``results/gaps.json``.

Usage:
    python scripts/run_all.py
"""

from __future__ import annotations

import csv
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SRC))

from orasr import ORASRRouter, ReasoningPath  # noqa: E402
from orasr.constants import DEFAULT_RANDOM_SEED  # noqa: E402

SEED = DEFAULT_RANDOM_SEED  # 42, sourced from the package constant


# --------------------------------------------------------------------------
# 1. Structural verification (deterministic, host-independent)
# --------------------------------------------------------------------------
def verify_structure() -> Dict[str, Any]:
    """Verify the routing structure directly against the committed engine."""
    router = ORASRRouter(enable_fast_path=True, enable_audit=False)

    thresholds = {
        "fast_path_threshold": router.FAST_PATH_THRESHOLD,
        "safe_path_threshold": router.SAFE_PATH_THRESHOLD,
        "fast_threshold_expected_0_3": router.FAST_PATH_THRESHOLD == 0.3,
        "safe_threshold_expected_0_7": router.SAFE_PATH_THRESHOLD == 0.7,
    }

    # Gate counts per pathway, read straight from the engine configuration.
    gate_counts = {
        path.name: len(router.pathways[path].gates)
        for path in (ReasoningPath.FAST, ReasoningPath.NORMAL, ReasoningPath.SAFE)
    }
    monotonic = gate_counts["FAST"] < gate_counts["NORMAL"] < gate_counts["SAFE"]

    # Gate inclusion is a strict superset chain (monotonic inclusion, not just count).
    fast_gates = set(router.pathways[ReasoningPath.FAST].gates)
    normal_gates = set(router.pathways[ReasoningPath.NORMAL].gates)
    safe_gates = set(router.pathways[ReasoningPath.SAFE].gates)
    inclusion_chain = fast_gates < normal_gates < safe_gates

    # Threshold partition: probe representative risk scores.
    def selected(risk: float) -> str:
        return router._select_pathway(risk, require_approval=False).name

    partition_probe = {
        "rho_0.00": selected(0.00),
        "rho_0.29": selected(0.29),
        "rho_0.30": selected(0.30),
        "rho_0.50": selected(0.50),
        "rho_0.69": selected(0.69),
        "rho_0.70": selected(0.70),
        "rho_1.00": selected(1.00),
    }
    partition_ok = (
        partition_probe["rho_0.00"] == "FAST"
        and partition_probe["rho_0.29"] == "FAST"
        and partition_probe["rho_0.30"] == "NORMAL"
        and partition_probe["rho_0.50"] == "NORMAL"
        and partition_probe["rho_0.69"] == "NORMAL"
        and partition_probe["rho_0.70"] == "SAFE"
        and partition_probe["rho_1.00"] == "SAFE"
    )

    # Non-bypass: a failed precondition gate must keep the action from executing.
    sentinel = {"ran": False}

    def tracking_action(_data: Any) -> str:
        sentinel["ran"] = True
        return "executed"

    bad = router.route(action=tracking_action, input_data=None, risk_score=0.1)
    non_bypass_ok = (not bad.safe) and (bad.action_result is None) and (not sentinel["ran"])

    return {
        "thresholds": thresholds,
        "gate_counts": gate_counts,
        "monotonic_gate_count": monotonic,
        "monotonic_gate_inclusion_chain": inclusion_chain,
        "partition_probe": partition_probe,
        "partition_correct": partition_ok,
        "non_bypass_holds": non_bypass_ok,
        "all_structural_checks_pass": bool(
            thresholds["fast_threshold_expected_0_3"]
            and thresholds["safe_threshold_expected_0_7"]
            and monotonic
            and inclusion_chain
            and partition_ok
            and non_bypass_ok
        ),
    }


# --------------------------------------------------------------------------
# 2. Synthetic cohort routing (deterministic structure, measured latency)
# --------------------------------------------------------------------------
COHORT_SPEC = [
    ("FAST", 3500, 0.00, 0.30),
    ("NORMAL", 4500, 0.30, 0.70),
    ("SAFE", 2000, 0.70, 1.00),
]


def build_cohort(rng: random.Random) -> List[Dict[str, Any]]:
    """Build the stratified synthetic scenario cohort (10,000 scenarios)."""
    cohort: List[Dict[str, Any]] = []
    for stratum, n, lo, hi in COHORT_SPEC:
        for _ in range(n):
            cohort.append({"stratum": stratum, "risk": rng.uniform(lo, hi)})
    rng.shuffle(cohort)
    return cohort


def run_cohort() -> Dict[str, Any]:
    """Route the synthetic cohort through the real engine.

    Structure (pathway assignment, gate pass/fail) is deterministic given the
    seed; latency is the wall-clock time measured on this host.
    """
    rng = random.Random(SEED)
    cohort = build_cohort(rng)

    router = ORASRRouter(enable_fast_path=True, enable_audit=True)

    def identity_action(data: Any) -> Any:
        return data

    pathway_counts: Dict[str, int] = {"FAST": 0, "NORMAL": 0, "SAFE": 0}
    latencies_by_path: Dict[str, List[float]] = {"FAST": [], "NORMAL": [], "SAFE": []}
    passed = 0
    blocked = 0

    for scenario in cohort:
        # Safe pathway requires human approval; grant it so the scenario reflects
        # an approved high-risk action (matching the cohort design).
        result = router.route(
            action=identity_action,
            input_data={"payload": 1},
            risk_score=scenario["risk"],
            human_approved=True,
        )
        pathway_counts[result.path.name] += 1
        latencies_by_path[result.path.name].append(result.latency * 1000.0)  # ms
        if result.safe:
            passed += 1
        else:
            blocked += 1

    total = len(cohort)

    def summarise(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"n": 0, "mean_ms": None, "p95_ms": None}
        ordered = sorted(vals)
        idx = max(0, int(round(0.95 * (len(ordered) - 1))))
        return {
            "n": len(vals),
            "mean_ms": round(statistics.fmean(vals), 4),
            "p95_ms": round(ordered[idx], 4),
        }

    return {
        "total_scenarios": total,
        "pathway_distribution": pathway_counts,
        "pathway_distribution_pct": {
            k: round(100.0 * v / total, 2) for k, v in pathway_counts.items()
        },
        "gate_pass_rate_pct": round(100.0 * passed / total, 4),
        "blocked": blocked,
        "latency_measured_ms": {k: summarise(v) for k, v in latencies_by_path.items()},
        "latency_note": (
            "Latencies are wall-clock timings measured on THIS host (see "
            "environment block). They are environment-specific and are NOT the "
            "manuscript literals."
        ),
    }


# --------------------------------------------------------------------------
# 3. Reject-option evaluation (deterministic, seed-driven robustness cohort)
# --------------------------------------------------------------------------
# The clean cohort above passes every gate (100%), so it carries no risk signal
# to trade against coverage. To produce an HONEST reject-option curve we build a
# separate, explicitly-labelled robustness cohort in which a deterministic
# fraction of scenarios carry malformed input (a structurally invalid payload
# that the committed PRECONDITION/POSTCONDITION gates legitimately reject). The
# violation rate is therefore measured by the REAL engine, not assigned by us.
#
#   - "violation" == the engine returns safe == False for an auto-executed action
#     (a true gate-level rejection from the committed validators).
#   - The reject option is an abstention threshold tau on the risk score: any
#     scenario with risk >= tau is DEFERRED (not auto-executed) and removed from
#     coverage; scenarios with risk < tau are auto-executed and contribute to the
#     measured risk. Sweeping tau from 1.0 down to 0.0 traces coverage vs risk.
#   - The marked operating point is the ORASR SAFE threshold (0.7): everything the
#     router already sends to the SAFE pathway is the natural deferral set.
ROBUSTNESS_N = 6000           # size of the robustness cohort
ROBUSTNESS_MALFORMED_RATE = 0.18  # deterministic fraction with invalid payload


def build_robustness_cohort(rng: random.Random) -> List[Dict[str, Any]]:
    """Build a robustness cohort: risk in [0,1] with a deterministic malformed mix.

    Malformed scenarios receive a payload the committed gates reject (None / a
    non-dict), so the resulting violation is produced by the real validators.
    """
    cohort: List[Dict[str, Any]] = []
    for _ in range(ROBUSTNESS_N):
        risk = rng.random()
        malformed = rng.random() < ROBUSTNESS_MALFORMED_RATE
        payload = None if malformed else {"payload": 1}
        cohort.append({"risk": risk, "malformed": malformed, "payload": payload})
    return cohort


def run_reject_option() -> Dict[str, Any]:
    """Route the robustness cohort and trace a real risk-coverage curve.

    Coverage = fraction auto-executed (risk < tau); risk = violation rate among
    those auto-executed, where a violation is the engine's own safe == False.
    """
    rng = random.Random(SEED)
    cohort = build_robustness_cohort(rng)
    router = ORASRRouter(enable_fast_path=True, enable_audit=False)

    def identity_action(data: Any) -> Any:
        return data

    # Route every scenario once; record (risk, violation, pathway) from the engine.
    records: List[Dict[str, Any]] = []
    lane_total: Dict[str, int] = {"FAST": 0, "NORMAL": 0, "SAFE": 0}
    lane_violations: Dict[str, int] = {"FAST": 0, "NORMAL": 0, "SAFE": 0}
    for sc in cohort:
        result = router.route(
            action=identity_action,
            input_data=sc["payload"],
            risk_score=sc["risk"],
            human_approved=True,  # grant SAFE-path approval; isolate gate behaviour
        )
        violation = not result.safe
        lane = result.path.name
        records.append({"risk": sc["risk"], "violation": violation, "lane": lane})
        lane_total[lane] += 1
        if violation:
            lane_violations[lane] += 1

    total = len(records)
    base_violation_rate = sum(1 for r in records if r["violation"]) / total

    # Risk-coverage sweep: tau from 1.0 (cover all) down to 0.0 (cover none).
    taus = [round(1.0 - i / 100.0, 2) for i in range(0, 101)]
    curve: List[Dict[str, Any]] = []
    for tau in taus:
        covered = [r for r in records if r["risk"] < tau]
        n_cov = len(covered)
        coverage = n_cov / total
        risk = (sum(1 for r in covered if r["violation"]) / n_cov) if n_cov else 0.0
        curve.append(
            {"tau": tau, "coverage": round(coverage, 6), "risk": round(risk, 6),
             "n_covered": n_cov}
        )

    # Operating point: ORASR SAFE threshold (0.7) used as the deferral cutoff.
    op_tau = router.SAFE_PATH_THRESHOLD
    op_covered = [r for r in records if r["risk"] < op_tau]
    op_coverage = len(op_covered) / total
    op_risk = (
        sum(1 for r in op_covered if r["violation"]) / len(op_covered)
        if op_covered else 0.0
    )

    lane_reliability = {
        lane: {
            "n": lane_total[lane],
            "violations": lane_violations[lane],
            "pass_rate_pct": round(
                100.0 * (lane_total[lane] - lane_violations[lane]) / lane_total[lane], 4
            ) if lane_total[lane] else None,
        }
        for lane in ("FAST", "NORMAL", "SAFE")
    }

    return {
        "robustness_cohort": {
            "n": total,
            "malformed_rate": ROBUSTNESS_MALFORMED_RATE,
            "base_violation_rate": round(base_violation_rate, 6),
            "note": (
                "Violations are produced by the committed PRECONDITION/POSTCONDITION "
                "validators rejecting a deterministic malformed-input fraction; they "
                "are measured by the real ORASRRouter, not assigned."
            ),
        },
        "risk_coverage_curve": curve,
        "operating_point": {
            "tau": op_tau,
            "coverage": round(op_coverage, 6),
            "risk": round(op_risk, 6),
            "note": "tau == ORASRRouter.SAFE_PATH_THRESHOLD (natural deferral set).",
        },
        "lane_reliability": lane_reliability,
    }


def environment_block() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "seed": SEED,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def gaps_block() -> Dict[str, Any]:
    return {
        "flat_monitor_gate_work_baseline": {
            "status": "REPRODUCED (gate-work / throughput)",
            "reason": (
                "scripts/flat_baseline.py instruments the committed SafetyGate.check "
                "to count gate evaluations per call for ORASR (adaptive) vs a flat "
                "monitor (all gates every call) on the seed-42 synthetic cohort. "
                "On that cohort ORASR=2.5 vs flat=4.0 evals/call -> 37.5% gate-work "
                "reduction and 1.60x work-normalised throughput. The same arithmetic "
                "on an assumed planning mix (58.1/31.5/10.4%) gives 1.94 evals/call; "
                "no MIMIC-IV data were used. Evaluation counts are "
                "host-INDEPENDENT integers."
            ),
        },
        "synthetic_gate_removal_ablation": {
            "status": "REPRODUCED",
            "reason": (
                "scripts/ablation.py runs progressive gate removal with a "
                "deterministic per-gate failure-injection suite (1,000 scenarios / "
                "gate, seed 42) on the SYNTHETIC cohort and reports coverage loss "
                "(undetected fault fraction) per configuration. Output "
                "results/ablation.csv|json. Each gate's removal leaks exactly its "
                "own fault type (coverage_loss 1.00); the full G1-G4 stack leaks 0."
            ),
        },
        "mimic_iv_retrospective_distribution": {
            "status": "REQUIRES CREDENTIALED DATA (runnable extraction provided)",
            "reason": (
                "scripts/extract_mimic.py reads a user-supplied MIMIC-IV copy, maps "
                "actions to risk via a transparent rule table, and emits the routing "
                "distribution -- but no MIMIC data is committed and the manuscript's "
                "8,412 actions @ 58.1/31.5/10.4% are NOT hardcoded. They must be "
                "regenerated by a user with credentialed PhysioNet access; absent "
                "data the script exits without fabricating."
            ),
        },
        "manuscript_latency_literals_and_mimic_latency": {
            "status": "ENVIRONMENT-SPECIFIC / REQUIRES CREDENTIALED DATA",
            "reason": (
                "Absolute wall-clock latency (per-pathway ms and any ms latency "
                "reduction or speedup) depends on host hardware/load and on the "
                "action mix. The gate-work reduction "
                "is reproduced as the portable invariant; absolute-ms reductions and "
                "any MIMIC-weighted latency / MIMIC-based ablation latency column "
                "remain host- and credentialed-data-specific and are NOT copied as "
                "literals here."
            ),
        },
        "gate_pass_rate_98_7": {
            "status": "NOT REPRODUCIBLE AS PUBLISHED",
            "reason": (
                "The committed gate validators pass for well-formed inputs, so a "
                "clean synthetic cohort yields a 100% pass rate. The 98.7% rate "
                "and the 130-block breakdown depend on a failure-injection / "
                "malformed-input mix that is not committed and is not "
                "reconstructed here (no tuning to match the paper)."
            ),
        },
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    structure = verify_structure()
    cohort = run_cohort()
    reject = run_reject_option()
    env = environment_block()
    gaps = gaps_block()

    # Reproducible empirical add-ons (host-independent gate-work + ablation).
    # Imported lazily so run_all stays usable even if a sibling script is absent.
    import flat_baseline  # noqa: E402
    import ablation  # noqa: E402

    flat = {"environment": env, **flat_baseline.run()}
    (RESULTS / "flat_baseline.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")

    abl = ablation.run()
    (RESULTS / "ablation.json").write_text(
        json.dumps(
            {"environment": env, "seed": abl["seed"], "n_per_gate": abl["n_per_gate"],
             "configurations": abl["configurations"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    with (RESULTS / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "active_gates", "fault_type", "n", "undetected", "coverage_loss"])
        for r in abl["rows"]:
            w.writerow([r["config"], r["active_gates"], r["fault_type"], r["n"],
                        r["undetected"], r["coverage_loss"]])

    # JSON artifacts.
    (RESULTS / "structural_verification.json").write_text(
        json.dumps(structure, indent=2), encoding="utf-8"
    )
    (RESULTS / "synthetic_cohort.json").write_text(
        json.dumps({"environment": env, **cohort}, indent=2), encoding="utf-8"
    )
    (RESULTS / "gaps.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    (RESULTS / "reject_option.json").write_text(
        json.dumps({"environment": env, **reject}, indent=2), encoding="utf-8"
    )

    # CSV: risk-coverage curve (consumed by the figure script).
    with (RESULTS / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tau", "coverage", "risk", "n_covered"])
        for pt in reject["risk_coverage_curve"]:
            writer.writerow([pt["tau"], pt["coverage"], pt["risk"], pt["n_covered"]])

    # CSV: per-lane gate reliability (consumed by the figure script).
    with (RESULTS / "lane_reliability.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lane", "n", "violations", "pass_rate_pct"])
        for lane in ("FAST", "NORMAL", "SAFE"):
            lr = reject["lane_reliability"][lane]
            writer.writerow([lane, lr["n"], lr["violations"], lr["pass_rate_pct"]])

    # CSV: pathway distribution (consumed by the figure script).
    with (RESULTS / "pathway_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pathway", "count", "pct", "mean_latency_ms", "p95_latency_ms"])
        for path in ("FAST", "NORMAL", "SAFE"):
            lat = cohort["latency_measured_ms"][path]
            writer.writerow(
                [
                    path,
                    cohort["pathway_distribution"][path],
                    cohort["pathway_distribution_pct"][path],
                    lat["mean_ms"],
                    lat["p95_ms"],
                ]
            )

    # CSV: threshold band map (consumed by the figure script).
    with (RESULTS / "threshold_bands.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pathway", "rho_low", "rho_high", "n_gates"])
        writer.writerow(["FAST", 0.0, structure["thresholds"]["fast_path_threshold"], 1])
        writer.writerow(
            [
                "NORMAL",
                structure["thresholds"]["fast_path_threshold"],
                structure["thresholds"]["safe_path_threshold"],
                3,
            ]
        )
        writer.writerow(["SAFE", structure["thresholds"]["safe_path_threshold"], 1.0, 4])

    print("Structural checks pass:", structure["all_structural_checks_pass"])
    print("Pathway distribution:", cohort["pathway_distribution"])
    print("Gate pass rate (clean cohort):", cohort["gate_pass_rate_pct"], "%")
    gw = flat["gate_work"]
    print(
        "Gate-work: ORASR=%.3f vs flat=%.1f evals/call -> %.2f%% reduction, %.2fx throughput"
        % (gw["orasr_evals_per_call"], gw["flat_evals_per_call"],
           gw["gate_work_reduction_pct"], gw["throughput_ratio_worknorm"])
    )
    print("Ablation full-stack overall coverage loss:",
          abl["configurations"]["FULL_G1G2G3G4"]["overall_coverage_loss"])
    print(
        "Reject-option base violation rate (robustness cohort):",
        round(100.0 * reject["robustness_cohort"]["base_violation_rate"], 2), "%",
    )
    print(
        "Operating point (tau=%.2f): coverage=%.3f risk=%.4f"
        % (
            reject["operating_point"]["tau"],
            reject["operating_point"]["coverage"],
            reject["operating_point"]["risk"],
        )
    )
    print("Measured latency (ms, host-specific):")
    for path in ("FAST", "NORMAL", "SAFE"):
        print("  ", path, cohort["latency_measured_ms"][path])
    # Supplementary deterministic analyses (seed 42). The 30-trial latency
    # variance run is host-specific and intentionally not called here.
    import manuscript_checks  # noqa: E402
    import detection_by_pathway  # noqa: E402
    import cost_model  # noqa: E402
    import misrouting_analysis  # noqa: E402
    import adversarial_shift  # noqa: E402
    import run_property_tests  # noqa: E402

    for mod in (manuscript_checks, detection_by_pathway, cost_model,
                misrouting_analysis, adversarial_shift):
        mod.main()
    run_property_tests.main()

    # v1.3.0 resubmission analyses (deterministic counts, seed 42).
    import os  # noqa: E402
    import subprocess  # noqa: E402
    import detection_clinical  # noqa: E402
    import ed_acuity_disagreement  # noqa: E402
    import structured_misrouting  # noqa: E402
    import upstream_reliability  # noqa: E402
    import baselines_extended  # noqa: E402
    import assurance_mechanisms  # noqa: E402
    import threshold_derivation  # noqa: E402
    import latency_realistic  # noqa: E402

    detection_clinical.main()
    ed_path = os.environ.get("MIMIC_ED_PATH", "D:/AVS/Datasets/mimic-ed-demo")
    demo_path = os.environ.get("MIMIC_DEMO_PATH", "D:/AVS/Datasets/mimic-iv-demo")
    py = [sys.executable, "-I"]
    if (Path(ed_path) / "ed" / "triage.csv.gz").exists():
        subprocess.run(py + [str(ROOT / "scripts" / "ed_acuity_disagreement.py"), "--ed-path", ed_path],
                       check=True)
    else:
        print("MIMIC-IV-ED demo not found; ed_acuity_disagreement skipped (E8 omitted)")
    if (Path(demo_path) / "hosp" / "prescriptions.csv.gz").exists():
        subprocess.run(py + [str(ROOT / "scripts" / "mimic_routing.py"), "--mimic-path", demo_path,
                             "--label", "demo", "--ed-path", ed_path], check=True)
    else:
        print("MIMIC-IV demo not found; mimic_routing (demo) skipped")
    full_path = os.environ.get("MIMIC_IV_PATH")
    if full_path and (Path(full_path) / "hosp" / "prescriptions.csv.gz").exists():
        subprocess.run(py + [str(ROOT / "scripts" / "mimic_routing.py"), "--mimic-path", full_path,
                             "--label", "full"], check=True)
    for mod in (structured_misrouting, upstream_reliability, baselines_extended,
                assurance_mechanisms, threshold_derivation):
        mod.main()
    # Host-specific timing (kept last).
    latency_realistic.main()
    print("Artifacts written to", RESULTS)


if __name__ == "__main__":
    main()
