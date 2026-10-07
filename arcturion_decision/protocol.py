"""Arcturion Decision Protocol v1. JSON only; no provider schema in callers."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import StrEnum
import hashlib
import json
import math
import re
import uuid

class Primitive(StrEnum):
    CHOICE = "CHOICE"
    SCORE = "SCORE"
    BOOLEAN_PROBABILITY = "BOOLEAN_PROBABILITY"
    RANK = "RANK"
    CONFIDENCE = "CONFIDENCE"
    ABSTAIN = "ABSTAIN"
    UTILITY = "UTILITY"
    RISK = "RISK"

class DecisionError(ValueError):
    pass

def encode(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError):
        raise DecisionError("INVALID_JSON") from None

def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()

def request_hashes(request):
    payload = asdict(request)
    hashes = {digest(payload)}
    # Preserve exact v1 receipt replay only when the new metadata is empty.
    if not payload.get("workflow"):
        payload.pop("workflow", None)
        hashes.add(digest(payload))
    return hashes

def finite(value, low=0, high=1):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high

SECRET = re.compile(r"(?i)(bearer\s+\S+|(?:sk|pk|ghp|xox[a-z])[_-][a-z0-9_-]{16,}|-----BEGIN .*PRIVATE KEY|(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+)")
SENSITIVE = re.compile(r"(?i)^(password|credential|secret|api[_-]?key|authorization|access[_-]?token|private[_-]?key)$")
BULK = {"conversation", "conversations", "transcript", "messages", "logs", "reasoning_trace", "chain_of_thought", "full_document"}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")

def safe(value, depth=0):
    if depth > 12:
        raise DecisionError("INPUT_TOO_DEEP")
    if isinstance(value, str):
        if SECRET.search(value):
            raise DecisionError("SENSITIVE_INPUT")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or SENSITIVE.fullmatch(key):
                raise DecisionError("SENSITIVE_INPUT")
            safe(key, depth + 1)
            safe(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            safe(item, depth + 1)
    elif value is not None and type(value) not in (int, float, bool):
        raise DecisionError("INVALID_JSON")
    encode(value)

@dataclass
class Request:
    state: dict
    question: str
    options: dict | list | None = None
    decision_type: str = "general"
    stakes: str = "low"
    required_confidence: float | None = None
    constraints: dict = field(default_factory=dict)
    requesting_agent: str = ""
    capacity: str = ""
    primitive: str = "CHOICE"
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    relevant_fields: list[str] | None = None
    multi_judge: bool = False
    requires_deliberation: bool = False
    workflow: dict = field(default_factory=dict)

    def validate(self):
        if self.stakes not in {"low", "moderate", "high"}:
            raise DecisionError("INVALID_STAKES")
        if self.capacity not in {"business", "personal"}:
            raise DecisionError("INVALID_CAPACITY")
        if not isinstance(self.requesting_agent, str) or not ID.fullmatch(self.requesting_agent):
            raise DecisionError("INVALID_AGENT")
        if not isinstance(self.decision_id, str) or not ID.fullmatch(self.decision_id):
            raise DecisionError("INVALID_DECISION_ID")
        if not isinstance(self.decision_type, str) or not ID.fullmatch(self.decision_type):
            raise DecisionError("INVALID_DECISION_TYPE")
        if not isinstance(self.question, str) or not 1 <= len(self.question) <= 1500:
            raise DecisionError("INVALID_QUESTION")
        if not isinstance(self.constraints, dict) or not isinstance(self.state, dict):
            raise DecisionError("INVALID_STATE")
        if self.required_confidence is not None and not finite(self.required_confidence):
            raise DecisionError("INVALID_CONFIDENCE")
        if type(self.multi_judge) is not bool or type(self.requires_deliberation) is not bool:
            raise DecisionError("INVALID_FLAGS")
        try:
            self.primitive = Primitive(self.primitive)
        except ValueError:
            raise DecisionError("INVALID_PRIMITIVE") from None
        if self.primitive in {Primitive.CHOICE, Primitive.RANK}:
            if not isinstance(self.options, dict) or not 2 <= len(self.options) <= 26:
                raise DecisionError("INVALID_OPTIONS")
            if any(not isinstance(k, str) or not ID.fullmatch(k) or not isinstance(v, str) or not v for k, v in self.options.items()):
                raise DecisionError("INVALID_OPTIONS")
        if self.primitive in {Primitive.SCORE, Primitive.UTILITY, Primitive.RISK}:
            if not isinstance(self.options, list) or not 2 <= len(self.options) <= 10 or not all(isinstance(x, str) and x for x in self.options):
                raise DecisionError("INVALID_SCALE")
        if len(encode(asdict(self))) > 262144:
            raise DecisionError("REQUEST_TOO_LARGE")
        safe(asdict(self))
        return self

def build_context(request, max_bytes=12000):
    """Explicit field projection, never silent evidence truncation or an LLM summarizer."""
    source = request.state
    if request.relevant_fields is not None:
        fields = request.relevant_fields
        if not isinstance(fields, list) or not fields or not all(isinstance(k, str) for k in fields):
            raise DecisionError("INVALID_RELEVANT_FIELDS")
        missing = [k for k in fields if k not in source]
        if missing:
            raise DecisionError("MISSING_FIELDS:" + ",".join(missing))
        source = {k: source[k] for k in fields}
    def check_bulk(value):
        if isinstance(value, dict):
            if BULK.intersection(k.lower() for k in value):
                raise DecisionError("CONTEXT_SELECTION_REQUIRED")
            for item in value.values():
                check_bulk(item)
        elif isinstance(value, list):
            for item in value:
                check_bulk(item)
    check_bulk(source)
    if not source:
        raise DecisionError("INSUFFICIENT_INFORMATION")
    packet = {"state": source, "question": request.question, "options": request.options,
              "constraints": request.constraints, "primitive": request.primitive,
              "decision_type": request.decision_type, "stakes": request.stakes}
    if len(encode(packet)) > max_bytes:
        raise DecisionError("CONTEXT_BUDGET_EXCEEDED")
    return packet

@dataclass
class Judgment:
    decision: object = None
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    abstain: bool = False
    reason: str = "MODEL_JUDGMENT"
    confidence_kind: str = "uncalibrated"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    provider_called: bool = True
    receipt_id: str | None = None

    def validate(self, request):
        if not finite(self.confidence) or type(self.abstain) is not bool:
            raise DecisionError("INVALID_PROVIDER_CONFIDENCE")
        if not isinstance(self.reason, str) or len(self.reason) > 240:
            raise DecisionError("INVALID_PROVIDER_REASON")
        if not isinstance(self.scores, dict) or not all(isinstance(k,str) and finite(v, -1e12, 1e12) for k,v in self.scores.items()):
            raise DecisionError("INVALID_PROVIDER_SCORES")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise DecisionError("INVALID_USAGE")
        if self.cost_usd is not None and not finite(self.cost_usd, 0, 1e6):
            raise DecisionError("INVALID_COST")
        if type(self.provider_called) is not bool:
            raise DecisionError("INVALID_USAGE")
        if not self.abstain:
            p = request.primitive
            if p == Primitive.CHOICE and (not isinstance(self.decision, str) or self.decision not in request.options):
                raise DecisionError("INVALID_PROVIDER_CHOICE")
            if p == Primitive.RANK and (not isinstance(self.decision, list) or len(self.decision) != len(request.options) or set(self.decision) != set(request.options)):
                raise DecisionError("INVALID_PROVIDER_RANK")
            if p == Primitive.BOOLEAN_PROBABILITY and not finite(self.decision):
                raise DecisionError("INVALID_PROVIDER_PROBABILITY")
            if p in {Primitive.SCORE, Primitive.UTILITY, Primitive.RISK} and not finite(self.decision, 0, len(request.options)-1):
                raise DecisionError("INVALID_PROVIDER_SCORE")
            if p == Primitive.CONFIDENCE and not finite(self.decision):
                raise DecisionError("INVALID_PROVIDER_CONFIDENCE")
        safe(asdict(self))
        return self

@dataclass
class ProviderSpec:
    name: str
    tier: int
    model: str
    primitives: list[str]
    context_bytes: int
    local: bool
    enabled: bool = True
    availability: str = "configured"
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    latency_ms: float | None = None
    specialties: list[str] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)
    independent_group: str = ""
    # Adapter wiring (read by create_engine, ignored by the engine itself):
    #   adapter: "openai_compatible", "anthropic", or "package.module:factory"
    #   options: adapter-specific settings such as the NAMES of environment variables
    adapter: str = ""
    options: dict = field(default_factory=dict)
