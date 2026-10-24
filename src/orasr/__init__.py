"""
ORASR: Operational Reasoning-Action Safety Routing
===================================================

Safety routing framework for clinical decision-making.
"""

from .constants import FRAMEWORK_NAME, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = "Chatchai Tritham"

from .router import ORASRRouter, RoutingResult
from .pathways import ReasoningPath, PathwayConfig
from .gates import SafetyGate, GateResult, GateType
from .reasoning import ReasoningTrace, ReasoningStep
from .constraints import RoutingConstraint, ConstraintViolation
from .router import risk_is_valid, INVALID_RISK_VIOLATION
from .audit import HashChainAuditLog, verify_chain, to_fhir_auditevent
from .approval import ApprovalToken, ApprovalVerifier, issue_token, action_hash
from .staging import StagedAction, CompensableAction

__all__ = [
    "ORASRRouter",
    "RoutingResult",
    "ReasoningPath",
    "PathwayConfig",
    "SafetyGate",
    "GateResult",
    "GateType",
    "ReasoningTrace",
    "ReasoningStep",
    "RoutingConstraint",
    "ConstraintViolation",
    "FRAMEWORK_NAME",
    "risk_is_valid",
    "INVALID_RISK_VIOLATION",
    "HashChainAuditLog",
    "verify_chain",
    "to_fhir_auditevent",
    "ApprovalToken",
    "ApprovalVerifier",
    "issue_token",
    "action_hash",
    "StagedAction",
    "CompensableAction",
]
