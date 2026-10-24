"""Tests for the v1.3.0 extensions: fail-closed inputs, type floor, staged
execution, compensation, signed approval tokens and the hash-chained audit log."""

import json

import pytest

from orasr import (ApprovalVerifier, CompensableAction, HashChainAuditLog, ORASRRouter,
                   ReasoningPath, StagedAction, action_hash, issue_token, to_fhir_auditevent,
                   verify_chain)

KEY = b"test-key-not-for-production"


def test_high_alert_type_forces_safe_at_low_risk():
    r = ORASRRouter(high_alert_types={"insulin"}, enable_audit=False)
    res = r.route(action=lambda d: d, input_data={"a": 1}, risk_score=0.05,
                  action_type="insulin", human_approved=True)
    assert res.path is ReasoningPath.SAFE and res.safe
    res2 = r.route(action=lambda d: d, input_data={"a": 1}, risk_score=0.05,
                   action_type="acetaminophen", human_approved=True)
    assert res2.path is ReasoningPath.FAST


def test_uncertainty_aware_routing_lifts_near_threshold():
    r = ORASRRouter(uncertainty_k=1.0, enable_audit=False)
    res = r.route(action=lambda d: d, input_data={"a": 1}, risk_score=0.65,
                  risk_uncertainty=0.08, human_approved=True)
    assert res.path is ReasoningPath.SAFE
    assert abs(res.metadata["effective_risk"] - 0.73) < 1e-12


def test_staged_irreversible_action_not_committed_when_g4_fails():
    effects = []
    act = StagedAction(prepare=lambda d: None, commit=lambda staged: effects.append(staged) or "done")
    res = ORASRRouter(enable_audit=False).route(action=act, input_data={"a": 1},
                                                risk_score=0.9, human_approved=True)
    assert res.safe is False and effects == [] and res.metadata["committed"] is False


def test_irreversible_staged_action_checked_on_fast_pathway():
    effects = []
    act = StagedAction(prepare=lambda d: None, commit=lambda s: effects.append(s) or s)
    res = ORASRRouter(enable_audit=False).route(action=act, input_data={"a": 1},
                                                risk_score=0.1, human_approved=True)
    assert res.path is ReasoningPath.FAST and res.safe is False and effects == []
    ok = StagedAction(prepare=lambda d: {"o": d}, commit=lambda s: effects.append(s) or s)
    res2 = ORASRRouter(enable_audit=False).route(action=ok, input_data={"a": 1},
                                                 risk_score=0.1, human_approved=True)
    assert res2.safe and effects == [{"o": {"a": 1}}]


def test_staged_action_commits_when_g4_passes():
    effects = []
    act = StagedAction(prepare=lambda d: {"order": d}, commit=lambda s: effects.append(s) or s)
    res = ORASRRouter(enable_audit=False).route(action=act, input_data={"a": 1},
                                                risk_score=0.9, human_approved=True)
    assert res.safe and effects == [{"order": {"a": 1}}] and res.metadata["committed"] is True


def test_reversible_action_is_compensated_when_g4_fails():
    log = []
    act = CompensableAction(run=lambda d: log.append("placed") or None,
                            compensate=lambda out: log.append("discontinued"))
    res = ORASRRouter(enable_audit=False).route(action=act, input_data={"a": 1},
                                                risk_score=0.9, human_approved=True)
    assert res.safe is False and log == ["placed", "discontinued"] and res.metadata["compensated"]


def _verifier(clock):
    return ApprovalVerifier(key=KEY, allowed_roles={"insulin": {"attending_physician", "pharmacist"}},
                            clock=clock)


def test_valid_token_is_accepted_once_and_replay_rejected():
    now = [1000.0]
    r = ORASRRouter(approval_verifier=_verifier(lambda: now[0]), enable_audit=False)
    x = {"dose_u": 4}
    tok = issue_token(KEY, "dr.a", "pharmacist", action_hash("insulin", x), ttl_s=60, now=1000.0)
    ok = r.route(action=lambda d: d, input_data=x, risk_score=0.9, action_type="insulin",
                 approval_token=tok)
    assert ok.safe and ok.human_approved
    replay = r.route(action=lambda d: d, input_data=x, risk_score=0.9, action_type="insulin",
                     approval_token=tok)
    assert not replay.safe and any("replay" in v for v in replay.violations)


@pytest.mark.parametrize("case", ["boolean", "bad_sig", "wrong_role", "other_action", "expired"])
def test_invalid_tokens_are_rejected(case):
    now = [1000.0]
    v = _verifier(lambda: now[0])
    r = ORASRRouter(approval_verifier=v, enable_audit=False)
    x = {"dose_u": 4}
    role = "nurse" if case == "wrong_role" else "pharmacist"
    h = action_hash("insulin", {"dose_u": 40} if case == "other_action" else x)
    tok = issue_token(b"wrong-key" if case == "bad_sig" else KEY, "dr.a", role, h, ttl_s=60, now=1000.0)
    if case == "expired":
        now[0] = 1100.0
    kwargs = {"human_approved": True} if case == "boolean" else {"approval_token": tok}
    res = r.route(action=lambda d: d, input_data=x, risk_score=0.9, action_type="insulin", **kwargs)
    assert res.safe is False and res.action_result is None


def test_audit_chain_detects_edit_delete_reorder_and_truncation(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = HashChainAuditLog(path)
    r = ORASRRouter(audit_log=log, enable_audit=False)
    for i in range(20):
        r.route(action=lambda d: d, input_data={"i": i}, risk_score=i / 20, human_approved=True)
    head = log.head()
    ok, n, h, probs = verify_chain(path, anchored_head=head)
    assert ok and n == 20 and h == head
    lines = path.read_text(encoding="utf-8").splitlines()

    def check(new_lines):
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return verify_chain(p, anchored_head=head)[0]

    edited = json.loads(lines[5]); edited["body"]["safe"] = not edited["body"]["safe"]
    assert not check(lines[:5] + [json.dumps(edited)] + lines[6:])
    assert not check(lines[:5] + lines[6:])
    assert not check(lines[:5] + [lines[6], lines[5]] + lines[7:])
    assert not check(lines[:-1])          # truncation: needs the anchored head
    assert verify_chain(path)[0]          # untouched file still verifies


def test_audit_reopen_continues_chain(tmp_path):
    path = tmp_path / "a.jsonl"
    HashChainAuditLog(path).append({"x": 1})
    log2 = HashChainAuditLog(path)
    log2.append({"x": 2})
    assert verify_chain(path)[:2] == (True, 2)


def test_fhir_mapping_has_required_fields():
    ev = to_fhir_auditevent({"timestamp": "2026-01-01T00:00:00Z", "safe": False,
                             "violations": ["G2 failed"], "pathway": "SAFE", "risk_score": 0.8})
    assert ev["resourceType"] == "AuditEvent" and ev["outcome"] == "4" and ev["recorded"]
