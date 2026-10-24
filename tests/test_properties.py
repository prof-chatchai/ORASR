"""Property-based tests (hypothesis) for the routing guarantees.

P1  every rho selects exactly one pathway, in the correct band
P2  rho1 < rho2  =>  gates(path(rho1)) is a subset of gates(path(rho2))
P4  reasoning-trace length = 1 + number of pre-execution gates, also when BLOCKED
Remark 1  (v1.3.0) any invalid rho (negative, NaN, inf, > 1, None, non-numeric) -> SAFE and blocked

A mutation test swaps in a non-monotone gate table and asserts the P2 checker
finds a violation, so the P2 property is not vacuously true.

Settings are fixed (derandomize=True, no example database) so runs are
deterministic. COUNTS is read by scripts/run_property_tests.py.
"""

import math
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import logging

from hypothesis import HealthCheck, find, given, settings, strategies as st
from hypothesis.errors import NoSuchExample

from orasr import ORASRRouter, ReasoningPath
from orasr.gates import GateType

logging.disable(logging.CRITICAL)

MAX_EXAMPLES = 2000
SETTINGS = settings(max_examples=MAX_EXAMPLES, derandomize=True, database=None, deadline=None,
                    suppress_health_check=list(HealthCheck))

T1, T2 = ORASRRouter.FAST_PATH_THRESHOLD, ORASRRouter.SAFE_PATH_THRESHOLD
BOUNDARY = [0.0, 1.0, T1, T2,
            math.nextafter(T1, 0.0), math.nextafter(T1, 1.0),
            math.nextafter(T2, 0.0), math.nextafter(T2, 1.0)]
rho_unit = st.one_of(st.sampled_from(BOUNDARY), st.floats(min_value=0.0, max_value=1.0))

COUNTS = {"P1": 0, "P2": 0, "P4": 0, "remark1_negative": 0, "remark1_out_of_range": 0,
          "mutation_detected": None, "mutation_counterexample": None}

ROUTER = ORASRRouter(enable_audit=False)


def gates_of(router, rho):
    return set(router.pathways[router._select_pathway(rho, False)].gates)


def p2_violated(router, r1, r2):
    """True if the P2 inclusion property fails for the ordered pair."""
    lo, hi = min(r1, r2), max(r1, r2)
    return lo < hi and not gates_of(router, lo) <= gates_of(router, hi)


@SETTINGS
@given(rho_unit)
def test_p1_exactly_one_pathway_in_band(rho):
    COUNTS["P1"] += 1
    path = ROUTER._select_pathway(rho, False)
    expected = (ReasoningPath.FAST if rho < T1 else
                ReasoningPath.NORMAL if rho < T2 else ReasoningPath.SAFE)
    assert path is expected
    res = ROUTER.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho,
                       human_approved=True)
    assert res.path is expected
    assert sum(res.path is p for p in ReasoningPath) == 1


@SETTINGS
@given(rho_unit, rho_unit)
def test_p2_gate_set_inclusion(r1, r2):
    COUNTS["P2"] += 1
    assert not p2_violated(ROUTER, r1, r2)


@SETTINGS
@given(rho_unit, st.booleans(), st.booleans())
def test_p4_trace_length_including_blocked(rho, malformed, approved):
    COUNTS["P4"] += 1
    res = ROUTER.route(action=lambda d: d, input_data=None if malformed else {"a": 1},
                       risk_score=rho, human_approved=approved)
    pre = [g for g in ROUTER.pathways[res.path].gates if g != GateType.POSTCONDITION]
    assert len(res.reasoning_trace.steps) == 1 + len(pre)
    if malformed:
        assert res.safe is False and res.action_result is None


@SETTINGS
@given(st.floats(max_value=-1e-300, allow_nan=False, allow_infinity=True))
def test_remark1_negative_fails_closed_to_safe(rho):
    """v1.3.0: negative scores are invalid input -> SAFE pathway and blocked."""
    COUNTS["remark1_negative"] += 1
    assert ROUTER._select_pathway(rho, False) is ReasoningPath.SAFE
    res = ROUTER.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho, human_approved=True)
    assert res.path is ReasoningPath.SAFE and res.safe is False and res.action_result is None
    assert any(v.startswith("R0_RiskInput") for v in res.violations)


@SETTINGS
@given(st.one_of(st.just(float("nan")),
                 st.floats(min_value=math.nextafter(1.0, 2.0), allow_nan=False)))
def test_remark1_nan_or_above_one_routes_safe_and_g2_fails(rho):
    COUNTS["remark1_out_of_range"] += 1
    res = ROUTER.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho,
                       human_approved=True)
    assert res.path is ReasoningPath.SAFE
    assert res.safe is False
    assert any(v.startswith("R0_RiskInput") for v in res.violations)
    assert any(v.startswith("G2_RiskAssessment") for v in res.violations)


@SETTINGS
@given(st.one_of(st.none(), st.text(max_size=5), st.booleans(), st.just(float("inf")),
                 st.just(float("-inf")), st.lists(st.floats(0, 1), max_size=2)))
def test_fail_closed_on_missing_or_non_numeric_risk(rho):
    COUNTS["fail_closed_non_numeric"] = COUNTS.get("fail_closed_non_numeric", 0) + 1
    res = ROUTER.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho, human_approved=True)
    assert res.path is ReasoningPath.SAFE and res.safe is False and res.action_result is None
    assert any(v.startswith("R0_RiskInput") for v in res.violations)


@SETTINGS
@given(rho_unit, st.one_of(st.just(float("nan")), st.floats(max_value=-1e-9, allow_nan=False)))
def test_fail_closed_on_invalid_uncertainty(rho, sigma):
    COUNTS["fail_closed_bad_sigma"] = COUNTS.get("fail_closed_bad_sigma", 0) + 1
    res = ROUTER.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho,
                       risk_uncertainty=sigma, human_approved=True)
    assert res.path is ReasoningPath.SAFE and res.safe is False


@SETTINGS
@given(rho_unit, st.floats(0, 0.5))
def test_uncertainty_aware_routing_never_lowers_pathway(rho, sigma):
    COUNTS["uncertainty_monotone"] = COUNTS.get("uncertainty_monotone", 0) + 1
    order = {ReasoningPath.FAST: 0, ReasoningPath.NORMAL: 1, ReasoningPath.SAFE: 2}
    ua = ORASRRouter(enable_audit=False, uncertainty_k=1.0)
    base = ROUTER._select_pathway(rho, False)
    res = ua.route(action=lambda d: d, input_data={"a": 1}, risk_score=rho,
                   risk_uncertainty=sigma, human_approved=True)
    assert order[res.path] >= order[base]


def test_mutation_non_monotone_table_is_detected(monkeypatch):
    mutant = ORASRRouter(enable_audit=False)
    monkeypatch.setattr(mutant.pathways[ReasoningPath.NORMAL], "gates",
                        [GateType.PRECONDITION, GateType.CONSTRAINT_VALIDATION])
    monkeypatch.setattr(mutant.pathways[ReasoningPath.SAFE], "gates",
                        [GateType.PRECONDITION, GateType.RISK_ASSESSMENT, GateType.POSTCONDITION])
    try:
        pair = find(st.tuples(rho_unit, rho_unit), lambda p: p2_violated(mutant, *p),
                    settings=settings(max_examples=MAX_EXAMPLES, derandomize=True, database=None))
    except NoSuchExample:
        pair = None
    COUNTS["mutation_detected"] = pair is not None
    COUNTS["mutation_counterexample"] = list(pair) if pair else None
    assert pair is not None
    # The unmodified router must not be flagged by the same search.
    assert not p2_violated(ROUTER, *pair)
