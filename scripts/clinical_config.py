"""Clinically meaningful *demonstration* configuration of G2 and G3 (v1.3.0).

Addresses the observation that, with the shipped minimal predicates, the Normal
pathway detects no fault class that Fast misses. Here G2 and G3 carry
medication-safety checks of the kind found in CPOE dose-range and drug-
interaction checking:

G2  (risk assessment)  action-class risk ceiling (as in demo_config) AND a
    dose-range check: the single dose must not exceed the ceiling in
    DOSE_LIMITS for that drug and unit.
G3  (constraint validation) registered RoutingConstraint objects:
    * allergy      -- the drug's class is not in the patient's allergy list;
    * interaction  -- no listed interacting class among active medications;
    * duplicate    -- the same ingredient is not already active;
    * time limit   -- caller-reported elapsed time <= 2 s (as before).

Gate selection followed a small hazard analysis (hazard -> control -> gate ->
fault test), recorded in HAZARD_TABLE and in results/detection_clinical.json.

Every table below is ILLUSTRATIVE. Dose ceilings are round, conservative
upper bounds on a typical adult single dose; the interaction list holds four
well-known class pairs. They are not a validated formulary and must be
replaced by an institutional drug knowledge base before any clinical use.
Drugs absent from DOSE_LIMITS are not dose-checked (reported as coverage).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orasr import GateType, ORASRRouter, SafetyGate  # noqa: E402
from orasr.constraints import RoutingConstraint, time_limit_constraint  # noqa: E402
from demo_config import CLASS_CEILINGS, MAX_ELAPSED_S, ceiling_for  # noqa: E402

# (ingredient keyword, unit) -> maximum single dose (illustrative)
DOSE_LIMITS: Dict[Tuple[str, str], float] = {
    ("acetaminophen", "mg"): 1000, ("ibuprofen", "mg"): 800, ("ketorolac", "mg"): 30,
    ("aspirin", "mg"): 650, ("morphine", "mg"): 30, ("hydromorphone", "mg"): 8,
    ("oxycodone", "mg"): 30, ("fentanyl", "mcg"): 200, ("lorazepam", "mg"): 4,
    ("ondansetron", "mg"): 24, ("metoprolol", "mg"): 200, ("furosemide", "mg"): 200,
    ("potassium chloride", "meq"): 40, ("insulin", "unit"): 100, ("heparin", "unit"): 10000,
    ("enoxaparin", "mg"): 150, ("warfarin", "mg"): 10, ("vancomycin", "mg"): 3000,
    ("prednisone", "mg"): 100, ("levothyroxine", "mcg"): 300, ("docusate", "mg"): 250,
    ("senna", "mg"): 17.2, ("pantoprazole", "mg"): 80, ("atorvastatin", "mg"): 80,
    ("amlodipine", "mg"): 10, ("lisinopril", "mg"): 80, ("gabapentin", "mg"): 1200,
    ("magnesium sulfate", "gm"): 4, ("sodium chloride 0.9%", "ml"): 2000,
}

DRUG_CLASSES: Dict[str, Tuple[str, ...]] = {
    "penicillin": ("amoxicillin", "ampicillin", "piperacillin", "penicillin", "nafcillin"),
    "nsaid": ("ibuprofen", "ketorolac", "naproxen", "aspirin", "celecoxib"),
    "sulfonamide": ("sulfamethoxazole",),
    "opioid": ("morphine", "hydromorphone", "oxycodone", "fentanyl", "methadone", "tramadol"),
    "benzodiazepine": ("lorazepam", "diazepam", "midazolam", "alprazolam", "clonazepam"),
    "anticoagulant": ("warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban"),
    "potassium": ("potassium chloride",),
    "k_sparing_diuretic": ("spironolactone", "amiloride", "triamterene"),
}

INTERACTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("anticoagulant", "nsaid", "bleeding"),
    ("anticoagulant", "sulfonamide", "warfarin potentiation / bleeding"),
    ("opioid", "benzodiazepine", "respiratory depression"),
    ("potassium", "k_sparing_diuretic", "hyperkalaemia"),
)

HAZARD_TABLE = [
    {"hazard": "malformed or missing order payload", "control": "schema check", "gate": "G1",
     "fault_test": "G1 input_data=None"},
    {"hazard": "overdose (unit or decimal error)", "control": "dose-range ceiling", "gate": "G2",
     "fault_test": "G2_dose dose x10"},
    {"hazard": "risk above the declared action class", "control": "class risk ceiling", "gate": "G2",
     "fault_test": "G2_ceiling"},
    {"hazard": "drug given despite documented allergy", "control": "allergy-class check", "gate": "G3",
     "fault_test": "G3_allergy"},
    {"hazard": "dangerous drug-drug combination", "control": "class-pair interaction check",
     "gate": "G3", "fault_test": "G3_interaction"},
    {"hazard": "duplicate therapy (same ingredient active)", "control": "duplicate-order check",
     "gate": "G3", "fault_test": "G3_duplicate"},
    {"hazard": "invalid or empty action output", "control": "output check", "gate": "G4",
     "fault_test": "G4 result=None"},
]

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def ingredient_of(drug: Any) -> Optional[str]:
    d = norm(drug)
    best = None
    for (ing, _u) in DOSE_LIMITS:
        if re.search(r"\b" + re.escape(ing) + r"\b", d) and (best is None or len(ing) > len(best)):
            best = ing
    if best is None:
        for members in DRUG_CLASSES.values():
            for m in members:
                if re.search(r"\b" + re.escape(m) + r"\b", d):
                    return m
    return best


def classes_of(drug: Any) -> List[str]:
    d = norm(drug)
    return [c for c, members in DRUG_CLASSES.items()
            if any(re.search(r"\b" + re.escape(m) + r"\b", d) for m in members)]


def parse_dose(value: Any) -> Optional[float]:
    """First number in a dose field ('1-2' -> 1.0); None when absent."""
    m = _NUM.search(str(value or ""))
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def dose_limit(drug: Any, unit: Any) -> Optional[float]:
    ing = ingredient_of(drug)
    if ing is None:
        return None
    return DOSE_LIMITS.get((ing, norm(unit)))


def dose_ok(ctx: Dict[str, Any]) -> bool:
    """True when the dose is within range or the drug/unit is not covered."""
    if "drug" not in ctx:
        return True
    lim = dose_limit(ctx.get("drug"), ctx.get("unit"))
    if lim is None:
        return True
    dose = parse_dose(ctx.get("dose"))
    return dose is None or dose <= lim


def allergy_ok(ctx: Dict[str, Any]) -> bool:
    allergies = set(ctx.get("allergies") or ())
    return not (allergies & set(classes_of(ctx.get("drug"))))


def interaction_hits(drug: Any, active: Iterable[Any]) -> List[str]:
    mine = set(classes_of(drug))
    hits = []
    for other in active or ():
        theirs = set(classes_of(other))
        for a, b, why in INTERACTIONS:
            if (a in mine and b in theirs) or (b in mine and a in theirs):
                hits.append(why)
    return hits


def interaction_ok(ctx: Dict[str, Any]) -> bool:
    return not interaction_hits(ctx.get("drug"), ctx.get("active_meds"))


def duplicate_ok(ctx: Dict[str, Any]) -> bool:
    ing = ingredient_of(ctx.get("drug"))
    if ing is None:
        return True
    return all(ingredient_of(m) != ing for m in ctx.get("active_meds") or ())


def _g2_validator(ctx: Dict[str, Any]) -> bool:
    rho = ctx.get("risk_score", 0)
    if isinstance(rho, float) and math.isnan(rho):
        return False
    ceiling = ceiling_for(ctx.get("action_class", "critical"))
    return rho <= ceiling and dose_ok(ctx)


def build_clinical_router(**router_kwargs: Any) -> ORASRRouter:
    router = ORASRRouter(**router_kwargs)
    router.gates[GateType.RISK_ASSESSMENT] = SafetyGate(
        gate_type=GateType.RISK_ASSESSMENT, validator=_g2_validator, name="G2_RiskAssessment",
        description="Class risk ceiling and dose-range check (illustrative tables)")
    router.add_constraint(RoutingConstraint("allergy", allergy_ok, "No documented allergy class"))
    router.add_constraint(RoutingConstraint("interaction", interaction_ok,
                                            "No listed interacting class among active meds"))
    router.add_constraint(RoutingConstraint("duplicate", duplicate_ok,
                                            "Same ingredient not already active"))
    router.add_constraint(time_limit_constraint(MAX_ELAPSED_S))
    return router


__all__ = ["DOSE_LIMITS", "DRUG_CLASSES", "INTERACTIONS", "HAZARD_TABLE", "CLASS_CEILINGS",
           "build_clinical_router", "dose_ok", "allergy_ok", "interaction_ok", "duplicate_ok",
           "interaction_hits", "ingredient_of", "classes_of", "parse_dose", "dose_limit"]
