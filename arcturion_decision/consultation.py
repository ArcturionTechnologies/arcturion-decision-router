"""Scoped advisory consultation and disposition; never an action authorization gate.

Metadata belongs to the calling workflow. This is not a prose risk classifier or
a universal completion lock. A signed-in local agent is a trusted local caller;
receipt matching proves content/scope consistency, not caller authentication.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from .protocol import Request, Judgment, DecisionError, digest, encode, safe, request_hashes

HIGH_TYPES = frozenset({
    "financial", "trade", "legal", "medical", "security", "permissions", "payment",
    "external_send", "destructive", "resource_commitment", "strategic_tradeoff",
    "production_change", "security_change", "external_commitment", "irreversible_action",
})
CONSEQUENCES = frozenset({
    "resource_commitment", "strategic_tradeoff", "production_change",
    "security_change", "external_commitment", "irreversible_action",
})
EXEMPT_WORK = frozenset({"arithmetic", "formatting", "exact_lookup", "settled_execution"})

def classify(request):
    """Prefer explicit workflow metadata; raise risk, never lower it."""
    meta = request.workflow
    if not isinstance(meta, dict) or set(meta) - {"id", "consequences", "work_kind", "decision_required"}:
        raise DecisionError("INVALID_WORKFLOW_METADATA")
    if "id" in meta and (not isinstance(meta["id"], str) or not meta["id"] or len(meta["id"]) > 160):
        raise DecisionError("INVALID_WORKFLOW_METADATA")
    reasons = meta.get("consequences", [])
    if not isinstance(reasons, list) or any(not isinstance(x, str) or x not in CONSEQUENCES for x in reasons):
        raise DecisionError("INVALID_WORKFLOW_CONSEQUENCES")
    if "decision_required" in meta and type(meta["decision_required"]) is not bool:
        raise DecisionError("INVALID_WORKFLOW_METADATA")
    if "work_kind" in meta and meta["work_kind"] not in EXEMPT_WORK | {"judgment"}:
        raise DecisionError("INVALID_WORK_KIND")
    high = bool(reasons) or request.decision_type.lower() in HIGH_TYPES or request.stakes == "high"
    # Conflicting metadata cannot turn an explicit consequential decision into arithmetic.
    exempt = not high and meta.get("work_kind") in EXEMPT_WORK and not meta.get("decision_required")
    required = high or (not exempt and (meta.get("decision_required") is True or request.stakes == "moderate"))
    return {"required": required, "exempt": exempt, "stakes": "high" if high else request.stakes,
            "reasons": sorted(set(reasons) | ({request.decision_type.lower()} if request.decision_type.lower() in HIGH_TYPES else set())),
            "basis": "explicit_workflow_and_declared_category", "prose_classification_complete": False}

def normalize(request):
    request.validate()
    classification = classify(request)
    request.stakes = classification["stakes"]
    return classification

def verify_receipt(ledger, request, *, require_review=False):
    """Read-only verification against the exact current request. Never invokes a provider."""
    normalize(request)
    record = ledger.inspect(request.decision_id, request.requesting_agent, request.capacity)
    result = record["result"]
    if not result:
        raise DecisionError("DECISION_PENDING_RECONCILIATION")
    if record["request_hash"] not in request_hashes(request):
        raise DecisionError("RECEIPT_REQUEST_MISMATCH")
    if (result.get("requesting_agent") != request.requesting_agent.upper()
            or result.get("capacity") != request.capacity
            or result.get("decision_id") != request.decision_id
            or result.get("state_hash") != digest(request.state)):
        raise DecisionError("RECEIPT_SCOPE_MISMATCH")
    if result.get("authorized") is not False or result.get("advisory_only") is not True:
        raise DecisionError("INVALID_ADVISORY_RECEIPT")
    review = ledger.get_review(request.decision_id, request.requesting_agent, request.capacity)
    if require_review and review is None:
        raise DecisionError("EVIDENCE_BASED_REVIEW_REQUIRED")
    if review and review["request_hash"] != record["request_hash"]:
        raise DecisionError("REVIEW_REQUEST_MISMATCH")
    return {"verified": True, "decision_id": request.decision_id,
            "requesting_agent": result["requesting_agent"], "capacity": request.capacity,
            "request_hash": record["request_hash"], "state_hash": result["state_hash"],
            "provider_receipt_ids": [p["receipt_id"] for p in result.get("provider_results", []) if p.get("receipt_id")],
            "consultation_status": ("unavailable" if result["abstain"] and not any(not p.get("abstain", True) for p in result.get("provider_results", []))
                                    else "handoff" if result.get("handoff") else "advice"),
            "review": review, "authorized": False, "advisory_only": True}

def record_review(ledger, request, review):
    """Persist the responsible agent's evidence-based acceptance/rejection/deliberation.

    Field checks enforce an explicit rationale and evidence references; they cannot
    establish whether natural-language reasoning is sound. Review remains auditable.
    """
    proof = verify_receipt(ledger, request)
    safe(review)
    required = {"reviewer", "disposition", "recommendation", "rationale", "evidence_fields", "advice_assessment"}
    if not isinstance(review, dict) or set(review) != required or len(encode(review)) > 5000:
        raise DecisionError("INVALID_REVIEW")
    if review["reviewer"] != request.requesting_agent.upper():
        raise DecisionError("REVIEWER_SCOPE_MISMATCH")
    if review["disposition"] not in {"accept", "reject", "provisional", "deliberated"}:
        raise DecisionError("INVALID_DISPOSITION")
    for key in ("rationale", "advice_assessment"):
        if not isinstance(review[key], str) or len(review[key].strip()) < 20:
            raise DecisionError("EVIDENCE_BASED_RATIONALE_REQUIRED")
    fields = review["evidence_fields"]
    if not isinstance(fields, list) or not fields or any(not isinstance(f, str) or f not in request.state for f in fields):
        raise DecisionError("CURRENT_EVIDENCE_REFERENCES_REQUIRED")
    record = ledger.inspect(request.decision_id, request.requesting_agent, request.capacity)
    result = record["result"]
    recommendation = review["recommendation"]
    try:
        Judgment(decision=recommendation, confidence=0, provider_called=False).validate(request)
    except (DecisionError, TypeError, ValueError):
        raise DecisionError("INVALID_RECOMMENDATION") from None
    if result["abstain"] and review["disposition"] in {"accept", "reject"}:
        raise DecisionError("HANDOFF_DELIBERATION_OR_PROVISIONAL_REQUIRED")
    if not result["abstain"]:
        if review["disposition"] == "accept" and recommendation != result["decision"]:
            raise DecisionError("ACCEPTANCE_DIFFERS_FROM_ADVICE")
        if review["disposition"] == "reject" and recommendation == result["decision"]:
            raise DecisionError("REJECTION_MATCHES_ADVICE")
    if review["disposition"] == "deliberated" and not result.get("handoff"):
        raise DecisionError("DELIBERATION_HANDOFF_REQUIRED")
    if review["disposition"] == "deliberated" and not any(not p.get("abstain", True) for p in result.get("provider_results", [])):
        raise DecisionError("UNAVAILABLE_REQUIRES_PROVISIONAL_ANALYSIS")
    value = dict(review, request_hash=proof["request_hash"],
                 authorized=False, advisory_only=True, independent_corroboration=False)
    ledger.attach_review(request.decision_id, request.requesting_agent, request.capacity, value)
    return verify_receipt(ledger, request, require_review=True)

def check_workflow(engine, request, *, require_review=True):
    """For explicit decision-bearing workflow transitions only; preserve action guards."""
    classification = normalize(request)
    if not classification["required"]:
        return {"consultation_required": False, "classification": classification,
                "authorized": False, "advisory_only": True}
    proof = verify_receipt(engine.ledger, request, require_review=require_review)
    return dict(proof, consultation_required=True, classification=classification)
