"""Compact decision interface: JSON in, JSON out. No action execution."""
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from . import Request, DecisionError, create_engine
from .protocol import encode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=["decide","evaluate","score","rank","gate","inspect","outcome","metrics","providers","review","verify"])
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input",type=Path,help="Private JSON request file, otherwise stdin")
    inputs.add_argument("--input-json","--request-json",help="Short SANITIZED JSON only; visible in process lists and shell history")
    parser.add_argument("--config",type=Path,help="Registry file (default: ARCTURION_DECISION_CONFIG, else the bundled deterministic-only registry)")
    parser.add_argument("--state-dir",type=Path,help="Ledger folder (default: ARCTURION_DECISION_STATE, else ~/.local/state/arcturion-decision)")
    parser.add_argument("--agent")
    parser.add_argument("--capacity",choices=["business","personal"])
    parser.add_argument("--decision-id")
    parser.add_argument("--human",action="store_true",help="Compact readable telemetry")
    args = parser.parse_args()
    try:
        opts = {"config_path":args.config}
        if args.state_dir:
            opts["state_root"] = args.state_dir
        engine = create_engine(**opts)
        if args.operation == "providers":
            result = {"providers":[asdict(p) for p in engine.providers]}
        elif args.operation in {"verify", "review"}:
            from .consultation import verify_receipt, record_review
            raw = read_input(args.input, args.input_json)
            request = Request(**raw["request"])
            if args.operation == "review":
                result = record_review(engine.ledger, request, raw["review"])
            else:
                result = verify_receipt(engine.ledger, request, require_review=raw.get("require_review", False))
        elif args.operation in {"inspect","metrics","outcome"}:
            if not args.agent or not args.capacity:
                raise DecisionError("EXPLICIT_SCOPE_REQUIRED")
            if args.operation == "metrics":
                result = engine.ledger.metrics(args.agent,args.capacity)
            elif args.operation == "inspect":
                result = engine.ledger.inspect(args.decision_id,args.agent,args.capacity)
            else:
                raw = read_input(args.input, args.input_json)
                result = engine.ledger.attach_outcome(args.decision_id,args.agent,args.capacity,raw["outcome"],raw.get("action_taken"))
        else:
            raw = read_input(args.input, args.input_json)
            if "requesting_agent" not in raw or "capacity" not in raw:
                raise DecisionError("EXPLICIT_SCOPE_REQUIRED")
            primitive = {"score":"SCORE","rank":"RANK","gate":"BOOLEAN_PROBABILITY"}.get(args.operation)
            if primitive:
                raw["primitive"] = primitive
            result = engine.evaluate(Request(**raw))
        display = result.get("result",result)
        if args.human and isinstance(display,dict) and "route" in display:
            print("ARCTURION DECISION")
            print("Agent: " + display["requesting_agent"] + " / " + display["capacity"])
            print("Decision ID: " + display["decision_id"])
            print("Decision: " + str(display["decision"]))
            print("Route: " + " -> ".join(display["route"]))
            for opinion in display["provider_results"]:
                print("Opinion - " + opinion["provider"] + ": " +
                      ("unavailable (" + opinion["reason"] + ")" if opinion["abstain"] else
                       str(opinion["decision"]) + " / confidence " + str(opinion["confidence"])))
            print("Confidence: " + str(round(display["confidence"]*100,2)) + "% (" + display["confidence_kind"] + ")")
            print("Risk: " + display["risk"] + "; abstain: " + str(display["abstain"]))
            print("Escalation reason: " + str(display["escalation_reason"]))
            print("Usage: " + encode(display["usage"]).decode())
            print("Latency ms: " + str(display["latency_ms"]))
            print("Outcome: " + encode(result.get("outcome")).decode())
            print("Advisory only; authorization is always false.")
        else:
            print(encode(result).decode())
        return 0
    except DecisionError as exc:
        print(encode({"status":"error","reason":str(exc).split(":")[0]}).decode())
        return 2
    except Exception:
        print('{"status":"error","reason":"INTERNAL_ERROR"}')
        return 2

def read_input(path=None, direct=None):
    if direct is not None:
        raw = direct.encode("utf-8")
    elif path:
        with path.open("rb") as handle:
            raw = handle.read(262145)
    else:
        raw = sys.stdin.buffer.read(262145)
    if len(raw)>262144:
        raise DecisionError("REQUEST_TOO_LARGE")
    try:
        value = json.loads(raw)
        if not isinstance(value,dict):
            raise ValueError()
    except (ValueError,UnicodeError):
        raise DecisionError("INVALID_JSON") from None
    from .protocol import safe
    safe(value)
    return value


