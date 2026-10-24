"""
Tamper-Evident Audit Log
========================

Append-only JSON Lines log in which every record carries the SHA-256 hash of
its predecessor (a hash chain). Editing, deleting, inserting or reordering a
record breaks the chain and is reported by :func:`verify_chain`. Truncation of
the tail is detectable only against an externally anchored head hash (for
example a periodically published or countersigned value), which is why
:meth:`HashChainAuditLog.head` exists.

This is an engineering prototype: access control, retention, off-host
replication and time-stamping by a trusted authority remain deployment
responsibilities. Each record maps onto the FHIR R4 ``AuditEvent`` resource
(``recorded``, ``agent``, ``entity``, ``outcome``); see :func:`to_fhir_auditevent`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    """Deterministic JSON encoding used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def entry_hash(prev_hash: str, body: Dict[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode("ascii") + canonical(body)).hexdigest()


class HashChainAuditLog:
    """Append-only, hash-chained JSONL audit log (thread-safe)."""

    def __init__(self, path: os.PathLike, fsync: bool = False):
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._seq = 0
        self._head = GENESIS
        if self.path.exists() and self.path.stat().st_size > 0:
            ok, n, head, _ = verify_chain(self.path)
            if not ok:
                raise ValueError(f"existing audit log {self.path} fails verification")
            self._seq, self._head = n, head

    def head(self) -> str:
        """Hash of the latest record (anchor this externally to detect truncation)."""
        return self._head

    def append(self, body: Dict[str, Any]) -> str:
        with self._lock:
            record = {"seq": self._seq, "prev": self._head, "body": body}
            h = entry_hash(self._head, {"seq": self._seq, "body": body})
            record["hash"] = h
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                if self.fsync:
                    fh.flush()
                    os.fsync(fh.fileno())
            self._seq += 1
            self._head = h
            return h


def iter_records(path: os.PathLike) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def verify_chain(path: os.PathLike, anchored_head: Optional[str] = None):
    """Verify a log. Returns (ok, n_records, head_hash, problems)."""
    problems: List[str] = []
    prev = GENESIS
    n = 0
    try:
        for i, rec in enumerate(iter_records(path)):
            if rec.get("seq") != i:
                problems.append(f"record {i}: sequence number {rec.get('seq')} != {i}")
            if rec.get("prev") != prev:
                problems.append(f"record {i}: prev-hash mismatch")
            expect = entry_hash(prev, {"seq": rec.get("seq"), "body": rec.get("body")})
            if rec.get("hash") != expect:
                problems.append(f"record {i}: content hash mismatch")
            prev = rec.get("hash", "")
            n += 1
    except (json.JSONDecodeError, OSError) as exc:
        problems.append(f"unreadable log: {exc}")
    if anchored_head is not None and prev != anchored_head:
        problems.append("head hash differs from the anchored value (truncation or rewrite)")
    return (not problems, n, prev, problems)


def to_fhir_auditevent(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map one ORASR audit body onto a minimal FHIR R4 AuditEvent resource."""
    return {
        "resourceType": "AuditEvent",
        "type": {"system": "http://dicom.nema.org/resources/ontology/DCM", "code": "110100",
                 "display": "Application Activity"},
        "action": "E",
        "recorded": body.get("timestamp"),
        "outcome": "0" if body.get("safe") else "4",
        "outcomeDesc": "; ".join(body.get("violations", [])) or "executed",
        "agent": [{"who": {"display": body.get("approver") or "ORASR router"},
                   "requestor": body.get("approver") is not None}],
        "source": {"observer": {"display": "ORASR"}},
        "entity": [{"what": {"display": body.get("action_type") or "action"},
                    "detail": [{"type": "pathway", "valueString": str(body.get("pathway"))},
                               {"type": "risk_score", "valueString": str(body.get("risk_score"))},
                               {"type": "action_hash", "valueString": str(body.get("action_hash"))}]}],
    }
