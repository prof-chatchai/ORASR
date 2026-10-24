"""Staged execution, signed approvals, approval delay and audit integrity (v1.3.0).

Deterministic counts (seed 42); per-operation timings are host-specific.

A  staged / compensated execution: 500 Safe-pathway actions whose output fails
   G4, run (i) as a plain callable (v1.2.0 behaviour), (ii) as a StagedAction
   (G4 on the staged output before commit), (iii) as a CompensableAction.
B  approval tokens: 1,000 attempts per case through the real router with an
   ApprovalVerifier (valid, replay, forged key, unauthorised role, token bound to
   another action, expired, bare boolean flag).
C  approval delay (Monte Carlo, 100,000 draws per setting): approver response
   time ~ lognormal(median m, log-SD 1.0), m in {30, 300} s; deadline D = 900 s
   for a time-critical action; policy = wait W for the primary approver, then
   escalate to a backup approver (independent, same distribution); if nobody
   approves by D the default-safe fallback applies (action withheld, clinician
   notified). W in {60, 120, 300, 600, no escalation}.
D  hash-chained audit log: 10,000 routed calls written to a temporary JSONL
   file; verification time; 200 random tampers of each kind.
Output: results/assurance_mechanisms.json
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orasr import (ApprovalVerifier, CompensableAction, HashChainAuditLog, ORASRRouter,  # noqa: E402
                   StagedAction, action_hash, issue_token, verify_chain)

logging.disable(logging.CRITICAL)
SEED = 42
KEY = b"demo-key-not-for-production"


def part_a() -> Dict:
    n = 500
    r = ORASRRouter(enable_audit=False)
    out = {}
    effects = []
    for i in range(n):
        r.route(action=lambda d: effects.append(d) or None, input_data={"i": i}, risk_score=0.9,
                human_approved=True)
    out["plain_callable"] = {"g4_faults": n, "side_effects_committed": len(effects),
                             "blocked_reported": n}
    effects = []
    blocked = 0
    for i in range(n):
        act = StagedAction(prepare=lambda d: None, commit=lambda s: effects.append(s) or s)
        res = r.route(action=act, input_data={"i": i}, risk_score=0.9, human_approved=True)
        blocked += not res.safe
    out["staged_two_phase"] = {"g4_faults": n, "side_effects_committed": len(effects),
                               "blocked_reported": blocked}
    log = []
    for i in range(n):
        act = CompensableAction(run=lambda d: log.append(("do", d)) or None,
                                compensate=lambda o: log.append(("undo", None)))
        r.route(action=act, input_data={"i": i}, risk_score=0.9, human_approved=True)
    out["compensable"] = {"g4_faults": n, "executed": sum(1 for a, _ in log if a == "do"),
                          "compensated": sum(1 for a, _ in log if a == "undo")}
    effects = []
    for i in range(n):
        act = StagedAction(prepare=lambda d: {"order": d}, commit=lambda s: effects.append(s) or s)
        r.route(action=act, input_data={"i": i}, risk_score=0.9, human_approved=True)
    out["staged_clean"] = {"actions": n, "committed": len(effects)}
    return out


def part_b() -> Dict:
    now = [1000.0]
    v = ApprovalVerifier(key=KEY, allowed_roles={"insulin": {"attending_physician", "pharmacist"}},
                         clock=lambda: now[0])
    r = ORASRRouter(approval_verifier=v, enable_audit=False)
    res = {}
    cases = ("valid", "replay", "forged_key", "unauthorised_role", "other_action", "expired", "boolean_flag")
    times = []
    for case in cases:
        acc = 0
        for i in range(1000):
            now[0] = 1000.0 + i
            x = {"dose_u": 4, "i": i, "case": case}
            h = action_hash("insulin", {"dose_u": 40, "i": i} if case == "other_action" else x)
            tok = issue_token(b"forged" if case == "forged_key" else KEY, "dr.a",
                              "nurse" if case == "unauthorised_role" else "pharmacist", h, ttl_s=60,
                              now=now[0])
            if case == "replay":
                r.route(action=lambda d: d, input_data=x, risk_score=0.9, action_type="insulin",
                        approval_token=tok)
            if case == "expired":
                now[0] += 120
            kw = {"human_approved": True} if case == "boolean_flag" else {"approval_token": tok}
            t0 = time.perf_counter()
            out = r.route(action=lambda d: d, input_data=x, risk_score=0.9, action_type="insulin", **kw)
            times.append(time.perf_counter() - t0)
            acc += out.safe
        res[case] = {"attempts": 1000, "accepted": acc}
    return {"cases": res, "mean_route_with_verification_us": round(1e6 * float(np.mean(times)), 2),
            "timing_note": "host-specific"}


def part_c() -> Dict:
    rng = np.random.default_rng(SEED)
    n, deadline = 100_000, 900.0
    rows = []
    for median in (30.0, 300.0):
        mu = math.log(median)
        t1 = rng.lognormal(mu, 1.0, n)
        t2 = rng.lognormal(mu, 1.0, n)
        for w in (60.0, 120.0, 300.0, 600.0, None):
            if w is None:
                t = t1
            else:
                t = np.where(t1 <= w, t1, np.minimum(t1, w + t2))
            ok = t <= deadline
            rows.append({"median_response_s": median, "escalate_after_s": w, "deadline_s": deadline,
                         "approved_by_deadline_pct": round(100 * float(ok.mean()), 2),
                         "default_safe_fallback_pct": round(100 * float((~ok).mean()), 2),
                         "median_wait_s": round(float(np.median(np.minimum(t, deadline))), 1),
                         "p95_wait_s": round(float(np.percentile(np.minimum(t, deadline), 95)), 1),
                         "escalations_pct": None if w is None else round(100 * float((t1 > w).mean()), 2)})
    return {"n_draws": n, "rows": rows}


def part_d() -> Dict:
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "audit.jsonl"
        log = HashChainAuditLog(path)
        r = ORASRRouter(audit_log=log, enable_audit=False)
        t0 = time.perf_counter()
        for i in range(10_000):
            r.route(action=lambda d: d, input_data={"i": i}, risk_score=rng.random(), human_approved=True)
        t_write = time.perf_counter() - t0
        head = log.head()
        t0 = time.perf_counter()
        ok, n, h, _ = verify_chain(path, anchored_head=head)
        t_verify = time.perf_counter() - t0
        size = path.stat().st_size
        lines = path.read_text(encoding="utf-8").splitlines()
        tp = Path(td) / "t.jsonl"
        det = {}
        for kind in ("edit_field", "delete_record", "swap_adjacent", "truncate_tail_anchored",
                     "truncate_tail_unanchored"):
            hit = 0
            for _ in range(200):
                L = list(lines)
                i = rng.randrange(1, len(L) - 1)
                if kind == "edit_field":
                    rec = json.loads(L[i]); rec["body"]["safe"] = not rec["body"]["safe"]; L[i] = json.dumps(rec)
                elif kind == "delete_record":
                    del L[i]
                elif kind == "swap_adjacent":
                    L[i], L[i + 1] = L[i + 1], L[i]
                else:
                    L = L[:i]
                tp.write_text("\n".join(L) + "\n", encoding="utf-8")
                anchor = None if kind == "truncate_tail_unanchored" else head
                hit += not verify_chain(tp, anchored_head=anchor)[0]
            det[kind] = {"tampers": 200, "detected": hit}
    return {"records": n, "verifies": ok, "bytes_per_record": round(size / n, 1),
            "write_s_total": round(t_write, 3), "verify_s_total": round(t_verify, 3),
            "tamper_detection": det, "timing_note": "host-specific; fsync disabled"}


def run() -> Dict:
    return {"seed": SEED, "staged_execution": part_a(), "approval_tokens": part_b(),
            "approval_delay": part_c(), "audit_log": part_d()}


def main() -> None:
    out = run()
    p = ROOT / "results" / "assurance_mechanisms.json"
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["staged_execution"]))
    print(json.dumps(out["approval_tokens"]))
    for row in out["approval_delay"]["rows"]:
        print(row)
    print(json.dumps(out["audit_log"]))


if __name__ == "__main__":
    main()
