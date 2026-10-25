"""Regression tests for the v1.3.1 fixes raised in the second review."""
import pytest

from orasr import ApprovalVerifier, ORASRRouter, StagedAction, action_hash, issue_token

KEY = b"test-key"


def _verifier(now):
    return ApprovalVerifier(key=KEY, allowed_roles={"insulin": {"pharmacist"}}, clock=lambda: now[0])


def test_negative_uncertainty_weight_rejected():
    with pytest.raises(ValueError):
        ORASRRouter(uncertainty_k=-2.0)
    with pytest.raises(ValueError):
        ORASRRouter(uncertainty_k=float("nan"))


def test_strict_mode_refuses_boolean_approval():
    r = ORASRRouter(enable_audit=False, require_verified_approval=True)
    res = r.route(lambda d: d, {"a": 1}, 0.9, human_approved=True)
    assert not res.safe and res.metadata["approval_mode"] == "refused_unverified"
    legacy = ORASRRouter(enable_audit=False).route(lambda d: d, {"a": 1}, 0.9, human_approved=True)
    assert legacy.safe and legacy.metadata["approval_mode"] == "legacy_boolean"


def test_blocked_call_does_not_consume_nonce():
    now = [1000.0]
    r = ORASRRouter(enable_audit=False, approval_verifier=_verifier(now))
    x = {"dose_u": 4}
    tok = issue_token(KEY, "dr.a", "pharmacist", action_hash("insulin", x), ttl_s=60, now=now[0])
    blocked = r.route(lambda d: d, x, 0.9, action_type="insulin", approval_token=tok, max_risk=0.5)
    assert not blocked.safe and blocked.metadata["approval_mode"] == "not_evaluated"
    retry = r.route(lambda d: d, x, 0.9, action_type="insulin", approval_token=tok)
    assert retry.safe and retry.human_approved
    replay = r.route(lambda d: d, x, 0.9, action_type="insulin", approval_token=tok)
    assert not replay.safe


def test_canonical_hash_rejects_ambiguous_inputs():
    with pytest.raises(ValueError):
        action_hash("t", {1: "a"})
    with pytest.raises(ValueError):
        action_hash("t", {"x": float("nan")})
    with pytest.raises(ValueError):
        action_hash("t", {"x": {1, 2}})
    assert action_hash("t", {"b": 1, "a": [1, 2.5]}) == action_hash("t", {"a": [1, 2.5], "b": 1})


def test_non_canonical_input_cannot_be_approved():
    now = [1000.0]
    r = ORASRRouter(enable_audit=False, approval_verifier=_verifier(now))
    tok = issue_token(KEY, "dr.a", "pharmacist", "0" * 64, ttl_s=60, now=now[0])
    res = r.route(lambda d: d, {1: "a"}, 0.9, action_type="insulin", approval_token=tok)
    assert not res.safe


def test_clock_skew_allowance():
    now = [1000.0]
    x = {"a": 1}
    tok = issue_token(KEY, "dr.a", "pharmacist", action_hash("insulin", x), ttl_s=60, now=1002.0)
    strict = _verifier(now)
    assert not strict.verify(tok, "insulin", x)[0]
    lenient = ApprovalVerifier(key=KEY, allowed_roles={"insulin": {"pharmacist"}}, clock=lambda: now[0],
                               clock_skew_s=5.0)
    assert lenient.verify(tok, "insulin", x)[0]


def test_exception_in_commit_aborts_staged_effect():
    aborted = []

    def bad_commit(z):
        raise RuntimeError("pharmacy link down")

    act = StagedAction(prepare=lambda d: {"order": d}, commit=bad_commit, abort=lambda z: aborted.append(z))
    res = ORASRRouter(enable_audit=False).route(act, {"a": 1}, 0.9, human_approved=True)
    assert not res.safe and res.metadata["committed"] is False and len(aborted) == 1
