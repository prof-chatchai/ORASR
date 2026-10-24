"""
ORASR Router
============

Main routing engine for safety-aware action execution.
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import math
import numbers
import time
import logging

from .pathways import ReasoningPath, PathwayConfig
from .gates import SafetyGate, GateType, GateResult
from .reasoning import ReasoningTrace, ReasoningStep
from .constraints import RoutingConstraint
from .staging import StagedAction, CompensableAction
from .approval import action_hash as _action_hash

INVALID_RISK_VIOLATION = (
    "R0_RiskInput: risk score missing, non-numeric, NaN or outside [0,1] (fail-closed)")


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def risk_is_valid(value: Any) -> bool:
    """A risk score is valid iff it is a real number (not bool), finite and in [0, 1]."""
    if not _is_real(value):
        return False
    v = float(value)
    return math.isfinite(v) and 0.0 <= v <= 1.0


def uncertainty_is_valid(value: Any) -> bool:
    """A risk uncertainty (standard deviation) must be a finite, non-negative real."""
    return _is_real(value) and math.isfinite(float(value)) and float(value) >= 0.0

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of routing decision"""

    path: ReasoningPath
    action_result: Any
    gates_passed: List[str]
    reasoning_trace: ReasoningTrace
    safe: bool
    latency: float
    human_approved: bool
    violations: List[str]
    metadata: Dict[str, Any]


class ORASRRouter:
    """
    Operational Reasoning-Action Safety Router.

    Routes actions through appropriate reasoning pathways with
    safety gate validation.
    """

    # Risk thresholds for pathway selection
    FAST_PATH_THRESHOLD = 0.3
    SAFE_PATH_THRESHOLD = 0.7

    def __init__(
        self,
        enable_fast_path: bool = True,
        enable_audit: bool = True,
        default_path: ReasoningPath = ReasoningPath.NORMAL,
        uncertainty_k: float = 0.0,
        high_alert_types: Optional[Any] = None,
        approval_verifier: Optional[Any] = None,
        audit_log: Optional[Any] = None,
        staged_execution: bool = True,
    ):
        """
        Initialize ORASR router.

        Args:
            enable_fast_path: Enable fast path for low-risk actions
            enable_audit: Enable audit logging
            default_path: Default pathway when risk unknown
            uncertainty_k: routing uses rho + k * sigma when the caller passes
                ``risk_uncertainty=sigma`` (0 = ignore uncertainty)
            high_alert_types: action types always routed SAFE regardless of rho
            approval_verifier: if set, approval must be a signed token
                (``approval_token=``) accepted by this verifier
            audit_log: optional HashChainAuditLog for durable, tamper-evident records
            staged_execution: run G4 on the staged output of a StagedAction
                before commit (two-phase execution)
        """
        self.uncertainty_k = float(uncertainty_k)
        self.high_alert_types = frozenset(high_alert_types or ())
        self.approval_verifier = approval_verifier
        self.audit_log = audit_log
        self.staged_execution = staged_execution
        self.enable_fast_path = enable_fast_path
        self.enable_audit = enable_audit
        self.default_path = default_path

        # Initialize safety gates
        self.gates: Dict[GateType, SafetyGate] = {}
        self._initialize_gates()

        # Pathway configurations
        self.pathways: Dict[ReasoningPath, PathwayConfig] = {
            ReasoningPath.FAST: PathwayConfig(
                name="FAST",
                gates=[GateType.PRECONDITION],
                max_latency=0.01,
                requires_approval=False,
            ),
            ReasoningPath.NORMAL: PathwayConfig(
                name="NORMAL",
                gates=[
                    GateType.PRECONDITION,
                    GateType.RISK_ASSESSMENT,
                    GateType.CONSTRAINT_VALIDATION,
                ],
                max_latency=0.1,
                requires_approval=False,
            ),
            ReasoningPath.SAFE: PathwayConfig(
                name="SAFE",
                gates=[
                    GateType.PRECONDITION,
                    GateType.RISK_ASSESSMENT,
                    GateType.CONSTRAINT_VALIDATION,
                    GateType.POSTCONDITION,
                ],
                max_latency=0.5,
                requires_approval=True,
            ),
        }

        self.constraints: List[RoutingConstraint] = []
        self.routing_history: List[RoutingResult] = []

        logger.info(
            f"ORASRRouter initialized: fast_path={enable_fast_path}, " f"audit={enable_audit}"
        )

    def _initialize_gates(self):
        """Initialize default safety gates"""

        # G1: Pre-condition check
        def precondition_validator(context: Dict) -> bool:
            input_data = context.get("input_data")
            if input_data is None:
                return False
            # Validate input structure
            return isinstance(input_data, dict)

        self.gates[GateType.PRECONDITION] = SafetyGate(
            gate_type=GateType.PRECONDITION,
            validator=precondition_validator,
            name="G1_Precondition",
            description="Validate input preconditions",
        )

        # G2: Risk assessment
        def risk_validator(context: Dict) -> bool:
            risk_score = context.get("risk_score", 0)
            max_risk = context.get("max_risk", 1.0)
            return risk_score <= max_risk

        self.gates[GateType.RISK_ASSESSMENT] = SafetyGate(
            gate_type=GateType.RISK_ASSESSMENT,
            validator=risk_validator,
            name="G2_RiskAssessment",
            description="Assess action risk level",
        )

        # G3: Constraint validation
        def constraint_validator(context: Dict) -> bool:
            # Check all constraints
            for constraint in self.constraints:
                if not constraint.validate(context):
                    return False
            return True

        self.gates[GateType.CONSTRAINT_VALIDATION] = SafetyGate(
            gate_type=GateType.CONSTRAINT_VALIDATION,
            validator=constraint_validator,
            name="G3_ConstraintValidation",
            description="Validate operational constraints",
        )

        # G4: Post-condition check
        def postcondition_validator(context: Dict) -> bool:
            result = context.get("action_result")
            # Verify result is valid
            return result is not None

        self.gates[GateType.POSTCONDITION] = SafetyGate(
            gate_type=GateType.POSTCONDITION,
            validator=postcondition_validator,
            name="G4_Postcondition",
            description="Verify action postconditions",
        )

    def route(
        self,
        action: Callable,
        input_data: Dict[str, Any],
        risk_score: float,
        require_human_approval: bool = False,
        reasoning_trace: Optional[ReasoningTrace] = None,
        **kwargs,
    ) -> RoutingResult:
        """
        Route action through safety pathway.

        Args:
            action: Action to execute
            input_data: Input data for action
            risk_score: Risk score (0-1)
            require_human_approval: Force human approval
            reasoning_trace: Optional existing reasoning trace
            **kwargs: Additional context

        Returns:
            RoutingResult with execution details
        """
        start_time = time.time()

        # Initialize reasoning trace
        if reasoning_trace is None:
            reasoning_trace = ReasoningTrace()

        # Validate upstream inputs: anything unusable fails closed to SAFE and blocks.
        action_type = kwargs.get("action_type")
        sigma = kwargs.get("risk_uncertainty")
        invalid_risk = not risk_is_valid(risk_score)
        if sigma is not None and not uncertainty_is_valid(sigma):
            invalid_risk = True
        effective_risk = risk_score
        if not invalid_risk and sigma is not None and self.uncertainty_k:
            effective_risk = min(1.0, float(risk_score) + self.uncertainty_k * float(sigma))
        forced = bool(require_human_approval or invalid_risk or (
            action_type is not None and action_type in self.high_alert_types))

        # Select pathway
        pathway = self._select_pathway(effective_risk, forced)

        reasoning_trace.add_step(
            ReasoningStep(
                timestamp=time.time(),
                description=f"Selected pathway: {pathway.name}",
                gate="ROUTER",
                result="PASS",
                confidence=1.0,
                metadata={"risk_score": risk_score},
            )
        )

        # Get pathway configuration
        config = self.pathways[pathway]

        # Build context
        context = {"input_data": input_data, "risk_score": risk_score, "pathway": pathway, **kwargs}

        # Execute gates (skip POSTCONDITION - it's checked after action executes)
        gates_passed = []
        violations = []
        safe = True
        if invalid_risk:
            safe = False
            violations.append(INVALID_RISK_VIOLATION)

        for gate_type in config.gates:
            if gate_type == GateType.POSTCONDITION:
                continue  # Skip postcondition in initial gate check
            gate = self.gates[gate_type]
            gate_result = gate.check(context)

            reasoning_trace.add_step(
                ReasoningStep(
                    timestamp=time.time(),
                    description=f"Gate check: {gate.name}",
                    gate=gate.name,
                    result="PASS" if gate_result.passed else "FAIL",
                    confidence=gate_result.confidence,
                    metadata=gate_result.metadata,
                )
            )

            if gate_result.passed:
                gates_passed.append(gate.name)
            else:
                safe = False
                violations.append(f"{gate.name}: {gate_result.message}")
                logger.warning(f"Gate {gate.name} failed: {gate_result.message}")

        # Check human approval if required
        human_approved = False
        if config.requires_approval or require_human_approval:
            if self.approval_verifier is not None:
                ok, why = self.approval_verifier.verify(
                    kwargs.get("approval_token"), action_type, input_data)
                human_approved = ok
                if not ok:
                    safe = False
                    violations.append(f"Approval token rejected: {why}")
            else:
                # Legacy boolean flag (prototype); see orasr.approval for signed tokens.
                human_approved = kwargs.get("human_approved", False)
                if not human_approved:
                    safe = False
                    violations.append("Human approval required but not provided")

        # Execute action if safe
        action_result = None
        committed = None
        compensated = False
        # Two-phase execution runs G4 on the staged output before commit. It applies on
        # every pathway for irreversible actions (so a misrouted irreversible action still
        # gets the output check) and on pathways that include G4 for reversible ones.
        staged = (self.staged_execution and isinstance(action, StagedAction)
                  and (GateType.POSTCONDITION in config.gates or not action.reversible))
        if safe and staged:
            # Two-phase execution: G4 checks the staged output before commit.
            try:
                prepared = action.prepare(input_data)
                context["action_result"] = prepared
                post_gate = self.gates[GateType.POSTCONDITION]
                post_result = post_gate.check(context)
                if post_result.passed:
                    gates_passed.append(post_gate.name)
                    action_result = action.commit(prepared)
                    committed = True
                else:
                    safe = False
                    committed = False
                    violations.append(
                        f"Postcondition failed on staged output: {post_result.message}; not committed")
                    if action.abort is not None:
                        action.abort(prepared)
            except Exception as e:
                safe = False
                committed = False
                violations.append(f"Action execution failed: {e}")
        elif safe:
            try:
                action_result = action(input_data)

                # Post-condition check if in pathway
                if GateType.POSTCONDITION in config.gates:
                    context["action_result"] = action_result
                    post_gate = self.gates[GateType.POSTCONDITION]
                    post_result = post_gate.check(context)

                    if not post_result.passed:
                        safe = False
                        violations.append(f"Postcondition failed: {post_result.message}")
                        if isinstance(action, CompensableAction):
                            action.compensate(action_result)
                            compensated = True
                            violations.append("Compensating action executed")
                    else:
                        gates_passed.append(post_gate.name)

            except Exception as e:
                safe = False
                violations.append(f"Action execution failed: {e}")
                logger.error(f"Action execution error: {e}")

        latency = time.time() - start_time

        # Create result
        result = RoutingResult(
            path=pathway,
            action_result=action_result,
            gates_passed=gates_passed,
            reasoning_trace=reasoning_trace,
            safe=safe,
            latency=latency,
            human_approved=human_approved,
            violations=violations,
            metadata={
                "risk_score": risk_score,
                "pathway_config": config.name,
                "constraints_checked": len(self.constraints),
                "effective_risk": effective_risk,
                "invalid_risk": invalid_risk,
                "action_type": action_type,
                "committed": committed,
                "compensated": compensated,
            },
        )

        if self.audit_log is not None:
            self.audit_log.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pathway": pathway.name,
                "risk_score": risk_score if not invalid_risk else repr(risk_score),
                "effective_risk": None if invalid_risk else effective_risk,
                "action_type": action_type,
                "action_hash": _action_hash(action_type, input_data),
                "gates_passed": gates_passed,
                "violations": violations,
                "safe": safe,
                "approver": getattr(kwargs.get("approval_token"), "approver_id", None),
                "committed": committed,
                "compensated": compensated,
            })

        # Log to history
        if self.enable_audit:
            self.routing_history.append(result)

        logger.info(
            f"Routing complete: path={pathway.name}, safe={safe}, "
            f"latency={latency:.3f}s, gates_passed={len(gates_passed)}"
        )

        return result

    def _select_pathway(self, risk_score: float, require_approval: bool) -> ReasoningPath:
        """Select appropriate reasoning pathway"""

        if require_approval or not risk_is_valid(risk_score):
            return ReasoningPath.SAFE  # fail-closed for unusable scores

        if self.enable_fast_path and risk_score < self.FAST_PATH_THRESHOLD:
            return ReasoningPath.FAST
        elif risk_score < self.SAFE_PATH_THRESHOLD:
            return ReasoningPath.NORMAL
        else:
            return ReasoningPath.SAFE

    def add_constraint(self, constraint: RoutingConstraint):
        """Add routing constraint"""
        self.constraints.append(constraint)
        logger.info(f"Constraint added: {constraint.name}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {}

        total = len(self.routing_history)
        safe_count = sum(1 for r in self.routing_history if r.safe)

        pathway_counts = {}
        for r in self.routing_history:
            pathway_counts[r.path.name] = pathway_counts.get(r.path.name, 0) + 1

        latencies = [r.latency for r in self.routing_history]

        return {
            "total_routings": total,
            "safe_rate": safe_count / total if total > 0 else 0,
            "pathway_distribution": pathway_counts,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
        }
