"""
Approval Tokens
===============

A Safe-pathway approval is a signed statement, not a boolean. A token binds

* the approver identity and role,
* the SHA-256 hash of the exact action (type + canonical input),
* an issue time and an expiry time,
* a single-use nonce,

and is authenticated with HMAC-SHA256 under a key held by the approval
service. :class:`ApprovalVerifier` rejects a token whose signature, role,
action binding, validity window or nonce fails. Key management (HSM, rotation)
and identity proofing of the approver are deployment responsibilities; an
asymmetric signature (e.g. Ed25519) can replace HMAC without changing the
token fields.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple


def _check_canonical(obj: Any, path: str = "$") -> None:
    """Admit only JSON types: dict with str keys, list/tuple, str, bool, None,
    int and finite float. Anything else (non-string keys, sets, NaN/inf,
    arbitrary objects) raises ValueError, so two distinct inputs cannot share a
    serialisation (v1.3.0 used ``default=str``, under which {1: 'a'} and
    {'1': 'a'} collided)."""
    if obj is None or isinstance(obj, (bool, str)):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite number at {path}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_canonical(v, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string key {k!r} at {path}")
            _check_canonical(v, f"{path}.{k}")
        return
    raise ValueError(f"type {type(obj).__name__} at {path} is not JSON")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: JSON types only, sorted keys, no insignificant
    whitespace, UTF-8, finite numbers. Close to RFC 8785 (JCS) for the inputs
    used here; number formatting follows Python's repr, so a verifier written
    in another language must use the same rule."""
    _check_canonical(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def action_hash(action_type: Optional[str], input_data: Any) -> str:
    payload = canonical_json({"type": action_type, "input": input_data})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovalToken:
    approver_id: str
    role: str
    action_hash: str
    issued_at: float
    expires_at: float
    nonce: str
    signature: str = ""

    def signing_payload(self) -> bytes:
        return json.dumps([self.approver_id, self.role, self.action_hash, round(self.issued_at, 6),
                           round(self.expires_at, 6), self.nonce],
                          separators=(",", ":")).encode("utf-8")


def issue_token(key: bytes, approver_id: str, role: str, a_hash: str, ttl_s: float = 300.0,
                now: Optional[float] = None, nonce: Optional[str] = None) -> ApprovalToken:
    t = time.time() if now is None else now
    unsigned = ApprovalToken(approver_id, role, a_hash, t, t + ttl_s, nonce or secrets.token_hex(16))
    sig = hmac.new(key, unsigned.signing_payload(), hashlib.sha256).hexdigest()
    return ApprovalToken(unsigned.approver_id, unsigned.role, unsigned.action_hash,
                         unsigned.issued_at, unsigned.expires_at, unsigned.nonce, sig)


@dataclass
class ApprovalVerifier:
    """Verifies approval tokens; ``allowed_roles`` maps action type -> roles."""

    key: bytes
    allowed_roles: Dict[Optional[str], Set[str]]
    default_roles: Set[str] = field(default_factory=lambda: {"attending_physician"})
    clock: Callable[[], float] = time.time
    clock_skew_s: float = 0.0
    _seen: Set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def verify(self, token: Any, action_type: Optional[str], input_data: Any) -> Tuple[bool, str]:
        if not isinstance(token, ApprovalToken):
            return False, "approval is not a signed token"
        expect = hmac.new(self.key, token.signing_payload(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, token.signature):
            return False, "bad signature"
        roles = self.allowed_roles.get(action_type, self.default_roles)
        if token.role not in roles:
            return False, f"role {token.role!r} not authorized for {action_type!r}"
        try:
            bound = action_hash(action_type, input_data)
        except ValueError as e:
            return False, f"input not canonicalisable: {e}"
        if token.action_hash != bound:
            return False, "token bound to a different action"
        now = self.clock()
        skew = max(0.0, float(self.clock_skew_s))
        if now < token.issued_at - skew or now > token.expires_at + skew:
            return False, "token outside validity window"
        with self._lock:
            if token.nonce in self._seen:
                return False, "token already used (replay)"
            self._seen.add(token.nonce)
        return True, "ok"


def all_roles(mapping: Dict[Optional[str], Iterable[str]]) -> Dict[Optional[str], Set[str]]:
    return {k: set(v) for k, v in mapping.items()}
