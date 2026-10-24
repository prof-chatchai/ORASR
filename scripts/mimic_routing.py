"""Routing of real clinical orders from MIMIC-IV (demo or credentialed full) (v1.3.0).

One pipeline, two data sources:
  * PUBLIC  MIMIC-IV Clinical Database Demo v2.2 (100 patients, open access)
            -> results/mimic_demo_routing.json
  * CREDENTIALED full MIMIC-IV hosp module (same file layout)
            -> results/mimic_full_routing.json, with small-cell suppression:
               every count below MIN_CELL (10) is reported as "<10" and no
               row-level data is written (PhysioNet DUA).

Steps
  1. Read hosp/prescriptions.csv.gz (medication orders) and hosp/poe.csv.gz
     (provider order entry; medication orders are counted once, from
     prescriptions, so POE rows with order_type "Medications" are skipped for
     mapping v2).
  2. Score each order with two transparent mappings:
       v1 = RISK_RULES of scripts/extract_mimic.py as released in v1.2.0
            (substring matching; kept for comparison);
       v2 = word-boundary matching with an ISMP-style high-alert medication
            list (Safe band), a short low-risk list (Fast band) and order-type
            rules for POE.
  3. Route every order with the committed ORASRRouter selector and report the
     pathway distribution, gate work and approval load.
  4. Run the clinical G2/G3 configuration (scripts/clinical_config.py) on every
     prescription with its real dose and the patient's concurrently active
     prescriptions, and report how often each check fires on unmodified data
     (alert burden; whether a firing is an error cannot be decided from the data).
  5. Inject faults into real orders (dose x10 on dose-covered orders; duplicate
     and interaction partners added to the active list) and report detection by
     pathway.
Optionally (``--ed-path``) the MIMIC-IV-ED demo pyxis dispensing table is routed too.

Usage
  python -I scripts/mimic_routing.py --mimic-path D:/AVS/Datasets/mimic-iv-demo --label demo \
         --ed-path D:/AVS/Datasets/mimic-ed-demo
  MIMIC_IV_PATH=/path/to/mimiciv/3.1 python -I scripts/mimic_routing.py --label full
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orasr import ORASRRouter  # noqa: E402
from extract_mimic import score_action as score_v1  # noqa: E402
from clinical_config import (build_clinical_router, dose_limit, dose_ok,  # noqa: E402
                             duplicate_ok, ingredient_of, interaction_ok, parse_dose)

logging.disable(logging.CRITICAL)
SEED = 42
MIN_CELL_FULL = 10

HIGH_ALERT = (
    "insulin", "heparin", "enoxaparin", "warfarin", "apixaban", "rivaroxaban", "dabigatran",
    "argatroban", "bivalirudin", "fondaparinux", "alteplase", "fentanyl", "morphine",
    "hydromorphone", "oxycodone", "methadone", "meperidine", "potassium chloride",
    "potassium phosphate", "vasopressin", "norepinephrine", "epinephrine", "phenylephrine",
    "dopamine", "dobutamine", "milrinone", "nitroprusside", "propofol", "midazolam",
    "dexmedetomidine", "ketamine", "lorazepam", "vecuronium", "rocuronium", "cisatracurium",
    "succinylcholine", "magnesium sulfate", "amiodarone", "digoxin", "methotrexate",
    "cyclophosphamide", "carboplatin", "cisplatin", "vincristine", "tacrolimus",
    "sodium chloride 3%", "sodium chloride 23.4%", "glipizide", "glyburide",
)
LOW_RISK = (
    "docusate", "senna", "sennosides", "bisacodyl", "polyethylene glycol", "multivitamin",
    "artificial tear", "flush", "lidocaine jelly", "simethicone", "calcium carbonate",
    "cholecalciferol", "vitamin d", "thiamine", "folic acid", "melatonin", "lanolin",
    "white petrolatum", "sodium chloride nasal", "guaifenesin", "menthol", "bacitracin",
)
POE_HIGH = ("blood bank", "tpn", "critical care", "hemodialysis")
POE_LOW = ("general care", "nutrition", "nursing", "adt orders", "respiratory", "consults")


def _has(text: str, words) -> bool:
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text) for w in words)


def score_v2_med(drug: str) -> float:
    t = drug.lower()
    if _has(t, HIGH_ALERT):
        return 0.85
    if _has(t, LOW_RISK):
        return 0.15
    return 0.50


def score_v2_poe(order_type: str, transaction: str) -> float:
    t = order_type.lower()
    if _has(t, POE_HIGH):
        return 0.85
    if _has(t, POE_LOW):
        return 0.15
    return 0.50


def band(rho: float) -> str:
    return "FAST" if rho < 0.3 else ("NORMAL" if rho < 0.7 else "SAFE")


def read_csv(path: Path, limit: Optional[int] = None):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                return
            yield row


def ts(v: str) -> Optional[datetime]:
    try:
        return datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


class Suppressor:
    def __init__(self, min_cell: int):
        self.min_cell = min_cell

    def n(self, x: int):
        return x if (self.min_cell <= 1 or x == 0 or x >= self.min_cell) else f"<{self.min_cell}"

    def counts(self, d: Dict[str, int]) -> Dict[str, Any]:
        return {k: self.n(v) for k, v in d.items()}


def pct(d: Dict[str, int]) -> Dict[str, float]:
    tot = sum(d.values())
    return {k: round(100.0 * v / tot, 2) for k, v in d.items()} if tot else {}


def route_dist(router: ORASRRouter, risks: List[float]) -> Dict[str, int]:
    c = Counter(router._select_pathway(r, False).name for r in risks)
    return {p: c.get(p, 0) for p in ("FAST", "NORMAL", "SAFE")}


def workload(dist: Dict[str, int]) -> Dict[str, float]:
    n = sum(dist.values())
    g = (dist["FAST"] + 3 * dist["NORMAL"] + 4 * dist["SAFE"]) / n
    s = dist["SAFE"] / n
    return {"gates_per_call": round(g, 4), "approvals_per_1000": round(1000 * s, 1),
            "expected_time_s_c10ms_A30s": round(g * 0.01 + 30 * s, 3),
            "flat4_universal_approval_time_s": 30.04}


def clinical_on_real(rx: List[Dict[str, str]], sup: Suppressor) -> Dict[str, Any]:
    router = build_clinical_router(enable_audit=False)
    by_adm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rx:
        st = ts(r.get("starttime", ""))
        if st is None:
            continue
        sp = ts(r.get("stoptime", "")) or st + timedelta(hours=24)
        by_adm[r.get("hadm_id", "")].append({"drug": r.get("drug", ""), "dose": r.get("dose_val_rx", ""),
                                             "unit": r.get("dose_unit_rx", ""), "start": st, "stop": sp,
                                             "rho": score_v2_med(r.get("drug", ""))})
    fires = {p: Counter() for p in ("FAST", "NORMAL", "SAFE")}
    n_by_path = Counter()
    covered = 0
    scen = []
    for orders in by_adm.values():
        orders.sort(key=lambda o: o["start"])
        for i, o in enumerate(orders):
            active = [q["drug"] for j, q in enumerate(orders)
                      if j != i and q["start"] <= o["start"] < q["stop"]]
            ctx = {"drug": o["drug"], "dose": o["dose"], "unit": o["unit"], "active_meds": active,
                   "allergies": []}
            res = router.route(action=lambda d: {"ok": 1}, input_data={"order": o["drug"]},
                               risk_score=o["rho"], human_approved=True, **ctx)
            p = res.path.name
            n_by_path[p] += 1
            lim = dose_limit(o["drug"], o["unit"])
            covered += lim is not None and parse_dose(o["dose"]) is not None
            for v in res.violations:
                key = v.split(":")[0]
                fires[p][key] += 1
            if not res.safe:
                fires[p]["any_block"] += 1
            fires[p]["chk_dose_over_ceiling"] += not dose_ok(ctx)
            fires[p]["chk_duplicate_ingredient"] += not duplicate_ok(ctx)
            fires[p]["chk_interaction"] += not interaction_ok(ctx)
            scen.append((o, active, lim))
    # fault injection on real orders
    rng = random.Random(SEED)
    inj = {p: {f: {"injected": 0, "detected": 0} for f in ("dose_x10", "duplicate", "interaction")}
           for p in ("FAST", "NORMAL", "SAFE")}
    partners = {"anticoagulant": "ibuprofen", "nsaid": "warfarin", "opioid": "lorazepam",
                "benzodiazepine": "oxycodone", "potassium": "spironolactone"}
    from clinical_config import classes_of, interaction_hits
    sample = rng.sample(scen, min(len(scen), 3000))
    for o, active, lim in sample:
        base = {"drug": o["drug"], "dose": o["dose"], "unit": o["unit"], "active_meds": active,
                "allergies": []}
        clean = router.route(action=lambda d: {"ok": 1}, input_data={"o": 1}, risk_score=o["rho"],
                             human_approved=True, **base)
        if not clean.safe:
            continue  # only inject into orders that pass cleanly, so detection is attributable
        trials = []
        d0 = parse_dose(o["dose"])
        if lim is not None and d0 is not None and d0 > 0:
            trials.append(("dose_x10", dict(base, dose=str(d0 * 10))))
        if ingredient_of(o["drug"]):
            trials.append(("duplicate", dict(base, active_meds=active + [o["drug"]])))
        cls = [c for c in classes_of(o["drug"]) if c in partners]
        if cls:
            partner = partners[cls[0]]
            if interaction_hits(o["drug"], [partner]):
                trials.append(("interaction", dict(base, active_meds=active + [partner])))
        for name, ctx in trials:
            r = router.route(action=lambda d: {"ok": 1}, input_data={"o": 1}, risk_score=o["rho"],
                             human_approved=True, **ctx)
            cell = inj[r.path.name][name]
            cell["injected"] += 1
            cell["detected"] += not r.safe
    n_tot = sum(n_by_path.values())
    return {
        "n_prescriptions_routed": sup.n(n_tot),
        "n_by_pathway": sup.counts(dict(n_by_path)),
        "dose_check_coverage_pct": round(100.0 * covered / n_tot, 2) if n_tot else None,
        "checks_firing_on_unmodified_orders": {p: sup.counts(dict(c)) for p, c in fires.items()},
        "checks_firing_pct_of_pathway": {p: {k: round(100.0 * v / n_by_path[p], 2) for k, v in c.items()}
                                         for p, c in fires.items() if n_by_path[p]},
        "fault_injection_on_real_orders": {p: {f: {k: sup.n(v) for k, v in cell.items()}
                                               for f, cell in d.items()} for p, d in inj.items()},
        "note": ("Allergy data are not in the MIMIC-IV hosp prescriptions table, so the allergy check "
                 "cannot fire on real orders. Firings on unmodified orders are alert burden; many "
                 "duplicate firings are legitimate (e.g., PRN plus scheduled orders)."),
    }



# Rescue-medication triggers (in the spirit of trigger-tool methods for adverse
# drug events): an index order of an implicated class is "triggered" when a
# rescue drug for that class is ordered in the same admission within 24 h after
# the index order starts. A trigger is a surrogate for possible harm, not a
# confirmed adverse event.
TRIGGERS = {
    "insulin -> dextrose 50% or glucagon": (("insulin",), ("dextrose 50%", "glucagon")),
    "opioid -> naloxone": (("morphine", "hydromorphone", "oxycodone", "fentanyl", "methadone",
                            "tramadol", "meperidine"), ("naloxone",)),
    "benzodiazepine -> flumazenil": (("lorazepam", "midazolam", "diazepam", "alprazolam",
                                      "clonazepam"), ("flumazenil",)),
    "warfarin -> phytonadione": (("warfarin",), ("phytonadione", "vitamin k")),
    "heparin -> protamine": (("heparin",), ("protamine",)),
}


def trigger_analysis(rx: List[Dict[str, str]], sup: "Suppressor") -> Dict[str, Any]:
    by_adm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rx:
        st = ts(r.get("starttime", ""))
        if st is not None:
            by_adm[r.get("hadm_id", "")].append({"drug": r.get("drug", "").lower(), "start": st})
    router = ORASRRouter(enable_audit=False)
    out: Dict[str, Any] = {}
    tot = {"index": Counter(), "triggered": Counter()}
    tot_v1 = {"index": Counter(), "triggered": Counter()}
    for name, (implicated, rescue) in TRIGGERS.items():
        idx_c, trig_c, trig_v1 = Counter(), Counter(), Counter()
        idx_v1 = Counter()
        for orders in by_adm.values():
            rescues = [o["start"] for o in orders if _has(o["drug"], rescue)]
            for o in orders:
                if not _has(o["drug"], implicated) or _has(o["drug"], ("flush",)):
                    continue
                p2 = router._select_pathway(score_v2_med(o["drug"]), False).name
                p1 = router._select_pathway(score_v1((o["drug"],)), False).name
                idx_c[p2] += 1
                idx_v1[p1] += 1
                hit = any(o["start"] < t <= o["start"] + timedelta(hours=24) for t in rescues)
                if hit:
                    trig_c[p2] += 1
                    trig_v1[p1] += 1
        tot["index"].update(idx_c); tot["triggered"].update(trig_c)
        tot_v1["index"].update(idx_v1); tot_v1["triggered"].update(trig_v1)
        out[name] = {"index_orders_by_pathway_v2": sup.counts(dict(idx_c)),
                     "triggered_index_orders_by_pathway_v2": sup.counts(dict(trig_c))}
    n_trig = sum(tot["triggered"].values())
    n_trig_v1 = sum(tot_v1["triggered"].values())
    return {
        "per_trigger": out,
        "total_index_orders_v2": sup.counts(dict(tot["index"])),
        "total_triggered_v2": sup.counts(dict(tot["triggered"])),
        "triggered_on_safe_pct_v2": round(100.0 * tot["triggered"]["SAFE"] / n_trig, 2) if n_trig else None,
        "total_triggered_v1": sup.counts(dict(tot_v1["triggered"])),
        "triggered_on_safe_pct_v1": round(100.0 * tot_v1["triggered"]["SAFE"] / n_trig_v1, 2) if n_trig_v1 else None,
        "definition": "index order of implicated class followed within 24 h (same admission) by a rescue-drug order; heparin flushes excluded",
        "caveat": "Surrogate trigger, not a confirmed adverse drug event; rescue drugs are also given for other reasons.",
    }


def run(root: Path, label: str, ed_root: Optional[Path], limit: Optional[int]) -> Dict[str, Any]:
    sup = Suppressor(MIN_CELL_FULL if label == "full" else 1)
    router = ORASRRouter(enable_audit=False)
    rx = list(read_csv(root / "hosp" / "prescriptions.csv.gz", limit))
    poe = list(read_csv(root / "hosp" / "poe.csv.gz", limit)) if (root / "hosp" / "poe.csv.gz").exists() else []
    n_pat = n_adm = None
    if (root / "hosp" / "patients.csv.gz").exists():
        n_pat = sum(1 for _ in read_csv(root / "hosp" / "patients.csv.gz"))
    if (root / "hosp" / "admissions.csv.gz").exists():
        n_adm = sum(1 for _ in read_csv(root / "hosp" / "admissions.csv.gz"))
    # v1 (as released): both sources, every row
    v1 = [score_v1((r.get("drug"), r.get("route"), r.get("drug_type"))) for r in rx] + \
         [score_v1((r.get("order_type"), r.get("transaction_type"))) for r in poe]
    # v2: prescriptions + non-medication POE orders
    v2_rx = [score_v2_med(r.get("drug", "")) for r in rx]
    poe_nonmed = [r for r in poe if r.get("order_type", "") != "Medications"]
    v2_poe = [score_v2_poe(r.get("order_type", ""), r.get("transaction_type", "")) for r in poe_nonmed]
    v2 = v2_rx + v2_poe
    d1, d2 = route_dist(router, v1), route_dist(router, v2)
    # band changes for prescriptions between v1 and v2
    changes = Counter()
    moved_drugs = Counter()
    for r, a, b in zip(rx, v1[:len(rx)], v2_rx):
        if band(a) != band(b):
            changes[f"{band(a)}->{band(b)}"] += 1
            moved_drugs[(r.get("drug", "").strip().lower(), band(a), band(b))] += 1
    top_moved = [{"drug": k[0], "v1": k[1], "v2": k[2], "n": sup.n(v)}
                 for k, v in moved_drugs.most_common(12) if v >= sup.min_cell]
    top_safe = Counter(r.get("drug", "").strip().lower() for r, s in zip(rx, v2_rx) if s >= 0.7)
    out: Dict[str, Any] = {
        "label": label, "source_root_kind": "public demo" if label == "demo" else "credentialed (aggregate only)",
        "n_patients": sup.n(n_pat) if n_pat is not None else None,
        "n_admissions": sup.n(n_adm) if n_adm is not None else None,
        "n_prescriptions": sup.n(len(rx)), "n_poe": sup.n(len(poe)),
        "n_poe_non_medication": sup.n(len(poe_nonmed)),
        "mapping_v1": {"description": "extract_mimic.RISK_RULES as released (substring)",
                       "n_actions": sup.n(len(v1)), "pathway_counts": sup.counts(d1),
                       "pathway_pct": pct(d1), "workload": workload(d1)},
        "mapping_v2": {"description": "word-boundary high-alert list; meds from prescriptions, "
                                      "non-medication orders from POE",
                       "n_actions": sup.n(len(v2)), "pathway_counts": sup.counts(d2),
                       "pathway_pct": pct(d2), "workload": workload(d2),
                       "prescriptions_pathway_pct": pct(route_dist(router, v2_rx)),
                       "poe_nonmed_pathway_pct": pct(route_dist(router, v2_poe)),
                       "risk_value_counts": sup.counts({f"{k:.2f}": v for k, v in Counter(v2).items()}),
                       "top_safe_drugs": [{"drug": k, "n": sup.n(v)} for k, v in top_safe.most_common(10)
                                          if v >= sup.min_cell]},
        "v1_to_v2_band_changes_prescriptions": sup.counts(dict(changes)),
        "v1_to_v2_top_moved_drugs": top_moved,
        "clinical_gates_on_real_orders": clinical_on_real(rx, sup),
        "rescue_trigger_analysis": trigger_analysis(rx, sup),
        "routing_thresholds": {"fast_lt": router.FAST_PATH_THRESHOLD, "safe_ge": router.SAFE_PATH_THRESHOLD},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if ed_root is not None and (ed_root / "ed" / "pyxis.csv.gz").exists():
        px = list(read_csv(ed_root / "ed" / "pyxis.csv.gz"))
        dpx = route_dist(router, [score_v2_med(r.get("name", "")) for r in px])
        out["ed_demo_pyxis"] = {"n_dispensations": sup.n(len(px)), "pathway_counts": sup.counts(dpx),
                                "pathway_pct": pct(dpx), "workload": workload(dpx)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-path", default=os.environ.get("MIMIC_IV_PATH"))
    ap.add_argument("--label", choices=("demo", "full"), default=None)
    ap.add_argument("--ed-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.mimic_path or not (Path(args.mimic_path) / "hosp" / "prescriptions.csv.gz").exists():
        sys.stderr.write("MIMIC-IV hosp/prescriptions.csv.gz not found (use --mimic-path or MIMIC_IV_PATH). "
                         "Nothing written; no number is fabricated.\n")
        sys.exit(2)
    root = Path(args.mimic_path)
    label = args.label or ("demo" if "demo" in root.as_posix().lower() else "full")
    out = run(root, label, Path(args.ed_path) if args.ed_path else None, args.limit)
    path = ROOT / "results" / f"mimic_{label}_routing.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("label", "n_patients", "n_admissions", "n_prescriptions", "n_poe")}))
    for m in ("mapping_v1", "mapping_v2"):
        print(m, out[m]["pathway_counts"], out[m]["pathway_pct"], out[m]["workload"])
    print("changes", out["v1_to_v2_band_changes_prescriptions"], out["v1_to_v2_top_moved_drugs"][:5])
    cg = out["clinical_gates_on_real_orders"]
    print("clinical", cg["n_by_pathway"], cg["dose_check_coverage_pct"], cg["checks_firing_pct_of_pathway"])
    print("inject", cg["fault_injection_on_real_orders"])
    ta = out["rescue_trigger_analysis"]
    print("triggers", ta["total_index_orders_v2"], ta["total_triggered_v2"], ta["triggered_on_safe_pct_v2"],
          "v1", ta["total_triggered_v1"], ta["triggered_on_safe_pct_v1"])
    if "ed_demo_pyxis" in out:
        print("pyxis", out["ed_demo_pyxis"])
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
