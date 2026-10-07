"""Scoped existing-workflow receipts. No inference, sends, or action authority."""
from . import default_state_root
from .ledger import Ledger
from .protocol import Request, DecisionError, digest
from .consultation import verify_receipt, classify, CONSEQUENCES
import sqlite3

# Explicit classes only. Unclassified items are never guessed from prose.
# Hosts may pass their own set via approval_consultation(..., classes=...).
CONSEQUENTIAL_APPROVAL_CLASSES = frozenset({
    "owner-decision", "resource-budget", "money-movement",
    "production-change", "production_change", "security-change", "security_change",
    "external-commitment", "external_commitment", "external-send", "external_send",
    "spend", "payment", "destructive", "irreversible-action", "irreversible_action",
})

def approval_payload(agent, capacity, title, context, gate_class, proposed_commands=None):
    return {"agent": agent.upper(), "capacity": capacity, "title": title,
            "context": context, "gate_class": gate_class,
            "proposed_commands": list(proposed_commands or [])}

def workflow_binding(payload):
    return digest(payload)

def _required_consequences(gate_class):
    if gate_class == "owner-decision":
        return CONSEQUENCES
    if gate_class in {"production-change", "production_change"}:
        return {"production_change"}
    if gate_class in {"security-change", "security_change"}:
        return {"security_change"}
    if gate_class in {"external-commitment", "external_commitment", "external-send", "external_send"}:
        return {"external_commitment"}
    if gate_class in {"destructive", "irreversible-action", "irreversible_action"}:
        return {"irreversible_action"}
    return {"resource_commitment"}

_CATEGORY_CONSEQUENCE = {"financial": "resource_commitment", "payment": "resource_commitment",
    "trade": "resource_commitment", "security": "security_change", "permissions": "security_change",
    "external_send": "external_commitment", "destructive": "irreversible_action"}

def _verify(raw_request, agent, capacity, binding, ledger=None, required_consequences=CONSEQUENCES):
    if not isinstance(raw_request, dict):
        raise DecisionError("DECISION_REQUEST_REQUIRED")
    request = Request(**raw_request)
    if request.requesting_agent.upper() != agent.upper() or request.capacity != capacity:
        raise DecisionError("WORKFLOW_SCOPE_MISMATCH")
    if request.state.get("workflow_binding") != binding:
        raise DecisionError("WORKFLOW_BINDING_MISMATCH")
    # Workflow metadata is authoritative: high stakes alone is not a matching
    # category. Check without changing the request or accepting a new fingerprint.
    classification = classify(request)
    declared = set(request.workflow.get("consequences", []))
    declared.add(_CATEGORY_CONSEQUENCE.get(request.decision_type, request.decision_type))
    if not declared.intersection(required_consequences) or classification["stakes"] != "high":
        raise DecisionError("WORKFLOW_CONSEQUENCE_REQUIRED")
    return verify_receipt(ledger or Ledger(default_state_root()), request, require_review=True)

def _status(status, reason, proof=None):
    return {"status": status, "reason": reason,
            "recommendation_status": "consulted" if status == "verified" else "provisional",
            "proof": proof, "authorized": False, "advisory_only": True}

def approval_consultation(payload, decision_request=None, *, ledger=None, classes=None):
    """Annotate consequential approval requests; never obstruct asking a human."""
    if payload["gate_class"] not in (CONSEQUENTIAL_APPROVAL_CLASSES if classes is None else classes):
        return None
    if decision_request is None:
        return _status("missing", "Consultation missing; human escalation remains available.")
    try:
        proof = _verify(decision_request, payload["agent"], payload["capacity"], workflow_binding(payload), ledger, _required_consequences(payload["gate_class"]))
    except (ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error) as exc:
        reason = str(exc) if isinstance(exc, DecisionError) else type(exc).__name__
        return _status("invalid", reason)
    if proof["consultation_status"] == "unavailable" or proof["review"]["disposition"] == "provisional":
        return _status("provisional", "Consultation unavailable or agent assessment provisional.", proof)
    return _status("verified", "Matching reviewed advisory receipt; separate action approval still required.", proof)

def consultation_notice(status):
    proof = status.get("proof") or {}
    identity = " Receipt: " + proof["decision_id"] + "." if proof.get("decision_id") else ""
    return ("ArcturionDecision: " + status["status"] + "; recommendation " +
            status["recommendation_status"] + ". " + status["reason"] + identity +
            " Consultation is advisory, not action authorization.")

def deployment_consultation(manifest, *, ledger=None):
    """Opt-in gate for a new deployment judgment, not settled execution/rollback."""
    required = manifest.get("decision_required", False)
    if type(required) is not bool:
        raise DecisionError("INVALID_DECISION_REQUIRED")
    if not required:
        return None
    if not isinstance(manifest.get("requesting_agent"), str) or manifest.get("capacity") not in {"business", "personal"}:
        raise DecisionError("EXPLICIT_WORKFLOW_SCOPE_REQUIRED")
    payload = {k: v for k, v in manifest.items() if k != "decision_request"}
    proof = _verify(manifest.get("decision_request"), manifest["requesting_agent"], manifest["capacity"], workflow_binding(payload), ledger, {"production_change"})
    if proof["consultation_status"] == "unavailable" or proof["review"]["disposition"] == "provisional":
        raise DecisionError("PROVISIONAL_ANALYSIS_ONLY")
    return proof
