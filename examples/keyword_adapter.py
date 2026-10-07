"""A minimal custom adapter, to show the interface. No model, no network.

Registry entry:  "adapter": "examples.keyword_adapter:make"

It picks the CHOICE option whose description shares the most words with the state.
Real adapters do the same three things: build a request from `packet`, call
something, and return a Judgment (or abstain).
"""
import re
from arcturion_decision import Judgment


def words(text):
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def make(spec):
    def adapter(request, packet, timeout):
        if not isinstance(request.options, dict):
            return Judgment(abstain=True, reason="UNSUPPORTED_PRIMITIVE", provider_called=False, cost_usd=0)
        evidence = words(str(packet["state"]))
        overlap = {key: len(evidence & words(text)) for key, text in request.options.items()}
        best = max(overlap, key=lambda key: (overlap[key], key))
        total = sum(overlap.values())
        if total == 0:
            return Judgment(abstain=True, reason="NO_OVERLAP", provider_called=False, cost_usd=0)
        return Judgment(best, round(overlap[best] / total, 3), dict(overlap), confidence_kind="word_overlap",
                        input_tokens=0, output_tokens=0, cost_usd=0.0)
    return adapter
