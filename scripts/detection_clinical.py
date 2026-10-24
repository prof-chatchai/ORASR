"""Fault detection by pathway with clinically meaningful G2/G3 (v1.3.0).

Deterministic, seed 42, synthetic cohort of scripts/latency_benchmark. Every
action carries a medication order drawn from CLEAN_ORDERS (dose within range,
no allergy, interaction or duplicate). Seven fault classes are then injected,
500 per stratum and class; the router picks the pathway from the action's own
risk score. Three configurations are compared: shipped default, the v1.2.0
demonstration configuration (ceilings + time limit), and the clinical
configuration of scripts/clinical_config.py.

Output: results/detection_clinical.json
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import latency_benchmark as lb  # noqa: E402
from clinical_config import (DOSE_LIMITS, HAZARD_TABLE, INTERACTIONS,  # noqa: E402
                             build_clinical_router)
from demo_config import build_default_router, build_demo_router, lowest_admissible_class  # noqa: E402

logging.disable(logging.CRITICAL)

SEED = lb.SEED
N_PER_CELL = 500
STRATA = ("FAST", "NORMAL", "SAFE")
FAULTS = ("G1", "G2_dose", "G3_allergy", "G3_interaction", "G3_duplicate", "G4")

# Clean orders: (drug, unit, typical dose). All within DOSE_LIMITS.
CLEAN_ORDERS = [("acetaminophen", "mg", 650), ("ondansetron", "mg", 4), ("metoprolol", "mg", 25),
                ("furosemide", "mg", 40), ("pantoprazole", "mg", 40), ("atorvastatin", "mg", 40),
                ("heparin", "unit", 5000), ("insulin", "unit", 4), ("morphine", "mg", 2),
                ("warfarin", "mg", 5), ("potassium chloride", "meq", 20), ("lorazepam", "mg", 1),
                ("ibuprofen", "mg", 400), ("vancomycin", "mg", 1000)]
BENIGN_ACTIVE = ["docusate", "senna", "levothyroxine", "amlodipine"]
# For allergy faults the drug must belong to a class; for interaction faults a partner exists.
ALLERGY_ORDERS = {"morphine": "opioid", "ibuprofen": "nsaid", "heparin": "anticoagulant",
                  "warfarin": "anticoagulant", "lorazepam": "benzodiazepine",
                  "potassium chloride": "potassium"}
PARTNER = {"warfarin": "ibuprofen", "heparin": "ketorolac", "ibuprofen": "warfarin",
           "morphine": "lorazepam", "lorazepam": "oxycodone", "potassium chloride": "spironolactone"}


def clean_scenario(rho: float, rng: random.Random) -> Dict[str, Any]:
    drug, unit, dose = rng.choice(CLEAN_ORDERS)
    return {"risk": rho, "input_data": {"order": drug}, "action_class": lowest_admissible_class(rho),
            "elapsed_time": 0.0, "result_valid": True, "drug": drug, "unit": unit, "dose": dose,
            "allergies": ["sulfonamide"], "active_meds": [rng.choice(BENIGN_ACTIVE)]}


def inject(sc: Dict[str, Any], fault: str, rng: random.Random) -> Dict[str, Any]:
    sc = dict(sc)
    if fault == "G1":
        sc["input_data"] = None
    elif fault == "G2_dose":
        sc["dose"] = DOSE_LIMITS[(sc["drug"], sc["unit"])] * 10
    elif fault == "G3_allergy":
        drug = rng.choice(sorted(ALLERGY_ORDERS))
        unit, dose = next((u, d) for n, u, d in CLEAN_ORDERS if n == drug)
        sc.update(drug=drug, unit=unit, dose=dose, allergies=[ALLERGY_ORDERS[drug]])
    elif fault == "G3_interaction":
        drug = rng.choice(sorted(PARTNER))
        unit, dose = next((u, d) for n, u, d in CLEAN_ORDERS if n == drug)
        sc.update(drug=drug, unit=unit, dose=dose, active_meds=[PARTNER[drug]])
    elif fault == "G3_duplicate":
        sc["active_meds"] = list(sc["active_meds"]) + [sc["drug"]]
    elif fault == "G4":
        sc["result_valid"] = False
    return sc


def route(router, sc):
    valid = sc["result_valid"]
    return router.route(action=lambda d: {"ok": 1} if valid else None, input_data=sc["input_data"],
                        risk_score=sc["risk"], action_class=sc["action_class"],
                        elapsed_time=sc["elapsed_time"], human_approved=True, drug=sc["drug"],
                        unit=sc["unit"], dose=sc["dose"], allergies=sc["allergies"],
                        active_meds=sc["active_meds"])


def evaluate(factory: Callable, cohort, faults) -> Dict[str, Any]:
    router = factory(enable_audit=False)
    rng = random.Random(SEED + 1)
    clean_blocks = {p: 0 for p in STRATA}
    for c in cohort:
        res = route(router, clean_scenario(c["risk"], rng))
        clean_blocks[res.path.name] += not res.safe
    table = {p: {f: {"injected": 0, "detected": 0} for f in FAULTS} for p in STRATA}
    mismatch = 0
    for sc in faults:
        res = route(router, sc)
        mismatch += res.path.name != sc["stratum"]
        cell = table[res.path.name][sc["fault"]]
        cell["injected"] += 1
        cell["detected"] += not res.safe
    return {"clean_false_blocks_by_pathway": clean_blocks, "faults_by_pathway": table,
            "fault_classes_detected_by_pathway": {
                p: [f for f in FAULTS if table[p][f]["detected"] == table[p][f]["injected"]
                    and table[p][f]["injected"] > 0] for p in STRATA},
            "pathway_mismatches": mismatch}


def run() -> Dict[str, Any]:
    cohort = lb.build_cohort(random.Random(SEED))
    rng = random.Random(SEED)
    faults: List[Dict[str, Any]] = []
    for stratum in STRATA:
        members = [c["risk"] for c in cohort if c["stratum"] == stratum]
        for fault in FAULTS:
            for _ in range(N_PER_CELL):
                sc = inject(clean_scenario(rng.choice(members), rng), fault, rng)
                sc["stratum"], sc["fault"] = stratum, fault
                faults.append(sc)
    return {
        "seed": SEED, "n_per_stratum_and_class": N_PER_CELL, "n_faults_total": len(faults),
        "hazard_table": HAZARD_TABLE, "interactions": [list(i) for i in INTERACTIONS],
        "detected_definition": "route() returned safe == False with human_approved=True",
        "configurations": {
            "default": evaluate(build_default_router, cohort, faults),
            "demo_v120": evaluate(build_demo_router, cohort, faults),
            "clinical": evaluate(build_clinical_router, cohort, faults),
        },
    }


def main() -> None:
    out = run()
    path = ROOT / "results" / "detection_clinical.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, cfg in out["configurations"].items():
        print(k, cfg["fault_classes_detected_by_pathway"], "false blocks",
              cfg["clean_false_blocks_by_pathway"], "mismatch", cfg["pathway_mismatches"])
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
