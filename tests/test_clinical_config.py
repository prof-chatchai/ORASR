"""Tests for the clinical demonstration configuration (scripts/clinical_config.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clinical_config import (build_clinical_router, dose_ok, duplicate_ok,  # noqa: E402
                             ingredient_of, interaction_ok, parse_dose)
from orasr import ReasoningPath  # noqa: E402


def test_word_boundary_ingredients():
    assert ingredient_of("Atorvastatin") == "atorvastatin"
    assert ingredient_of("Insulin Glargine") == "insulin"
    assert ingredient_of("Nystatin Oral Suspension") is None


def test_dose_parsing_and_ceiling():
    assert parse_dose("1-2") == 1.0 and parse_dose("") is None
    assert dose_ok({"drug": "Acetaminophen", "dose": "650", "unit": "mg"})
    assert not dose_ok({"drug": "Acetaminophen", "dose": "6500", "unit": "mg"})
    assert dose_ok({"drug": "Unlisted drug", "dose": "99999", "unit": "mg"})


def test_interaction_and_duplicate():
    assert not interaction_ok({"drug": "Warfarin", "active_meds": ["Ibuprofen"]})
    assert interaction_ok({"drug": "Warfarin", "active_meds": ["Docusate"]})
    assert not duplicate_ok({"drug": "Heparin", "active_meds": ["Heparin Flush"]})


def test_normal_catches_what_fast_misses():
    r = build_clinical_router(enable_audit=False)
    ctx = dict(drug="Warfarin", dose="5", unit="mg", active_meds=["Ibuprofen"], allergies=[])
    fast = r.route(action=lambda d: {"ok": 1}, input_data={"o": 1}, risk_score=0.1,
                   human_approved=True, **ctx)
    normal = r.route(action=lambda d: {"ok": 1}, input_data={"o": 1}, risk_score=0.5,
                     human_approved=True, **ctx)
    assert fast.path is ReasoningPath.FAST and fast.safe
    assert normal.path is ReasoningPath.NORMAL and not normal.safe
