"""Acceptance tests: fixture adapters only. No real model, network or credential is touched."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcturion_decision.engine import Engine
from arcturion_decision.ledger import Ledger
from arcturion_decision.protocol import (
    DecisionError, Judgment, Primitive, ProviderSpec, Request, build_context, encode,
)
from arcturion_decision.providers import normalize, build_prompt, answer_shape, post_json

OPTIONS = {"BUY": "Buy", "SELL": "Sell", "HOLD": "Hold"}


class FixtureAdapter:
    """Records simulated calls and returns isolated copies of fixture judgments."""
    def __init__(self, judgment):
        self.judgment = judgment
        self.calls = []

    def __call__(self, request, packet, timeout):
        self.calls.append({"packet": deepcopy(packet), "timeout": timeout})
        if isinstance(self.judgment, Exception):
            raise self.judgment
        return deepcopy(self.judgment)


def spec(name, tier, *, local=False, **kwargs):
    return ProviderSpec(
        name=name, tier=tier, model="simulated-fixture-model",
        primitives=[p.value for p in Primitive], context_bytes=16000, local=local,
        input_usd_per_million=0.1, output_usd_per_million=0.2,
        independent_group=name, **kwargs,
    )


def judgment(choice="BUY", confidence=.95, **kwargs):
    return Judgment(choice, confidence, input_tokens=20, output_tokens=3,
                    cost_usd=0, **kwargs)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Ledger(Path(self.tmp.name) / "ledger")

    def engine(self, entries=(), **kwargs):
        providers = [entry[0] for entry in entries]
        adapters = {s.name: FixtureAdapter(j) for s, j in entries}
        self.adapters = adapters
        return Engine(providers, adapters, self.ledger, **kwargs)

    def request(self, **kwargs):
        values = dict(requesting_agent="BUILDER", capacity="business", state={"signal": 7}, question="Select the operational action",
                      options=OPTIONS.copy(), decision_id="fixture-decision")
        values.update(kwargs)
        return Request(**values)

    def test_a_deterministic_threshold_has_zero_model_calls(self):
        e = self.engine([(spec("fixture-local", 1, local=True), judgment())])
        r = e.evaluate(self.request(
            primitive="BOOLEAN_PROBABILITY", options=None,
            constraints={"deterministic": {"op": "threshold", "field": "signal",
                                          "operator": "gte", "value": 7}}))
        self.assertEqual(r["decision"], 1)
        self.assertEqual(r["usage"]["provider_calls"], 0)
        self.assertEqual(r["models_used"], [])
        self.assertEqual(r["route"], ["T0:deterministic"])
        self.assertEqual(r["confidence_kind"], "deterministic")
        self.assertEqual(self.adapters["fixture-local"].calls, [])

    def test_b_routine_local_terminates_before_hosted_or_reasoning(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), judgment()),
            (spec("fixture-hosted", 2), judgment()),
            (spec("fixture-reasoning", 4), judgment()),
        ])
        r = e.evaluate(self.request())
        self.assertEqual(r["models_used"], ["fixture-local"])
        self.assertFalse(r["abstain"])
        self.assertFalse(r["escalated"])
        self.assertEqual(r["usage"]["provider_calls"], 1)
        self.assertFalse(self.adapters["fixture-hosted"].calls)
        self.assertFalse(self.adapters["fixture-reasoning"].calls)
        self.assertFalse(r["authorized"])
        self.assertTrue(r["advisory_only"])

    def test_c_moderate_chooses_hosted_without_local_or_alternate(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), judgment()),
            (spec("fixture-hosted", 2), judgment()),
            (spec("fixture-alt", 3), judgment()),
        ])
        r = e.evaluate(self.request(stakes="moderate"))
        self.assertEqual(r["models_used"], ["fixture-hosted"])
        self.assertFalse(self.adapters["fixture-local"].calls)
        self.assertFalse(self.adapters["fixture-alt"].calls)

    def test_d_strong_disagreement_vetoes_weighted_majority(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), judgment("BUY", .95)),
            (spec("fixture-hosted", 2), judgment("BUY", .94)),
            (spec("fixture-alt", 3), judgment("SELL", .90)),
        ])
        # Direct stability isolates the veto from the bounded two-judge router.
        req = self.request().validate()
        values = [(s, deepcopy(e.adapters[s.name].judgment)) for s in e.providers]
        winner, agreement, unstable = Engine.stability(req, values)
        self.assertEqual(winner.decision, "BUY")
        self.assertTrue(unstable)
        r = self.engine([
            (spec("fixture-hosted", 2), judgment("BUY", .95)),
            (spec("fixture-alt", 3), judgment("SELL", .90)),
        ]).evaluate(self.request(stakes="high"))
        self.assertTrue(r["abstain"])
        self.assertEqual(r["reason"], "DECISION_UNSTABLE")
        self.assertEqual(r["escalation_reason"], "DECISION_UNSTABLE")
        self.assertEqual(len(r["escalation_packet"]["judgments"]), 2)

    def test_d_mission_weighted_consensus_and_instability_examples(self):
        req = self.request().validate()
        for confidences, choices, expected_unstable in [
            ([.91, .88, .84, .54], ["BUY", "BUY", "BUY", "HOLD"], False),
            ([.54, .51, .52, .48], ["BUY", "SELL", "BUY", "HOLD"], True),
        ]:
            with self.subTest(confidences=confidences):
                pairs = [(spec("fixture-" + str(i), 2), judgment(c, confidence))
                         for i, (c, confidence) in enumerate(zip(choices, confidences))]
                winner, agreement, unstable = Engine.stability(req, pairs)
                self.assertEqual(unstable, expected_unstable)
                self.assertEqual(winner.decision, "BUY")
                if not expected_unstable:
                    self.assertGreater(agreement, .8)

    def test_e_high_stakes_conflict_escalates_relevant_packet(self):
        e = self.engine([
            (spec("fixture-hosted", 2), judgment("BUY", .95)),
            (spec("fixture-alt", 3), judgment("SELL", .93)),
            (spec("fixture-reasoning", 4), judgment("HOLD", .96)),
        ])
        r = e.evaluate(self.request(stakes="high",
            state={"signal": 7, "irrelevant": "x" * 50000}, relevant_fields=["signal"]))
        self.assertEqual(r["decision"], "HOLD")
        self.assertTrue(r["escalated"])
        self.assertEqual(r["usage"]["provider_calls"], 3)
        packet = self.adapters["fixture-reasoning"].calls[0]["packet"]
        self.assertEqual(packet["state"], {"signal": 7})
        self.assertEqual(packet["unresolved"], "DECISION_UNSTABLE")
        self.assertEqual(len(packet["judgments"]), 2)
        self.assertLessEqual(len(encode(packet)), 6000)
        stored = self.ledger.inspect(r["decision_id"], "BUILDER", "business")
        self.assertNotIn("signal", json.dumps(stored["result"]))
        self.assertNotIn("escalation_packet", stored["result"])

    def test_e_novel_reasoning_skips_decision_providers(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), judgment()),
            (spec("fixture-hosted", 2), judgment()),
            (spec("fixture-reasoning", 4), judgment("HOLD")),
        ])
        r = e.evaluate(self.request(requires_deliberation=True))
        self.assertEqual(r["models_used"], ["fixture-reasoning"])
        self.assertEqual(r["escalation_reason"], "DELIBERATION_REQUIRED")
        self.assertEqual(self.adapters["fixture-reasoning"].calls[0]["packet"]["judgments"], [])

    def test_context_overflow_abstains_without_silent_truncation(self):
        e = self.engine([(spec("fixture-local", 1, local=True), judgment())],
                        context_bytes=500)
        r = e.evaluate(self.request(state={"evidence": "x" * 1000}))
        self.assertEqual(r["reason"], "CONTEXT_BUDGET_EXCEEDED")
        self.assertEqual(r["usage"]["provider_calls"], 0)
        self.assertTrue(r["missing_information"])

    def test_escalation_overflow_does_not_call_reasoning(self):
        e = self.engine([(spec("fixture-reasoning", 4), judgment())],
                        escalation_bytes=500)
        r = e.evaluate(self.request(state={"evidence": "x" * 1000},
                                   requires_deliberation=True))
        self.assertEqual(r["reason"], "ESCALATION_CONTEXT_REQUIRED")
        self.assertFalse(self.adapters["fixture-reasoning"].calls)

    def test_bulk_context_requires_explicit_projection(self):
        req = self.request(state={"signal": 7, "logs": ["unrelated"]}).validate()
        with self.assertRaisesRegex(DecisionError, "CONTEXT_SELECTION_REQUIRED"):
            build_context(req)
        req.relevant_fields = ["signal"]
        self.assertEqual(build_context(req)["state"], {"signal": 7})
        req.relevant_fields = ["missing"]
        with self.assertRaisesRegex(DecisionError, "MISSING_FIELDS"):
            build_context(req)

    def test_policy_block_and_failure_preempt_all_models(self):
        def failing_policy(request):
            raise RuntimeError("simulated policy outage")
        for policy, expected in [(lambda request: True, "POLICY_BLOCKED"),
                                 (failing_policy, "POLICY_UNAVAILABLE")]:
            with self.subTest(expected=expected):
                e = self.engine([(spec("fixture-local", 1, local=True), judgment())],
                                policy=policy)
                r = e.evaluate(self.request(decision_id=expected))
                self.assertEqual(r["reason"], expected)
                self.assertEqual(r["models_used"], [])
        for key in ["blocked", "approval_required"]:
            r = self.engine().evaluate(self.request(decision_id=key, constraints={key: True}))
            self.assertEqual(r["reason"], "POLICY_BLOCKED")

    def test_missing_required_evidence_identifies_field(self):
        r = self.engine().evaluate(self.request(constraints={"required_fields": ["budget"]}))
        self.assertEqual(r["reason"], "INSUFFICIENT_INFORMATION")
        self.assertEqual(r["missing_information"], ["budget"])

    def test_failed_local_falls_back_once_and_redacts_exception(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), RuntimeError("private provider failure detail")),
            (spec("fixture-hosted", 2), judgment()),
            (spec("fixture-reasoning", 4), judgment()),
        ])
        r = e.evaluate(self.request())
        self.assertEqual(r["models_used"], ["fixture-local", "fixture-hosted"])
        self.assertEqual(r["decision"], "BUY")
        self.assertEqual(len(self.adapters["fixture-local"].calls), 1)
        self.assertEqual(r["provider_results"][0]["reason"], "PROVIDER_ERROR")
        self.assertNotIn("private provider failure detail", json.dumps(r))
        self.assertEqual(r["usage"]["unknown_cost_calls"], 1)

    def test_call_limit_prevents_reasoning_and_never_retries(self):
        e = self.engine([
            (spec("fixture-local", 1, local=True), judgment(confidence=.2)),
            (spec("fixture-hosted", 2), judgment(confidence=.2)),
            (spec("fixture-alt", 3), judgment(confidence=.2)),
            (spec("fixture-reasoning", 4), judgment()),
        ], max_calls=2)
        r = e.evaluate(self.request())
        self.assertTrue(r["abstain"])
        self.assertEqual(r["usage"]["provider_calls"], 2)
        self.assertIn("BUDGET:call_limit", r["route"])
        self.assertTrue(all(len(a.calls) <= 1 for a in self.adapters.values()))

    def test_zero_cost_budget_blocks_hosted_and_unknown_price(self):
        for unknown in [False, True]:
            with self.subTest(unknown=unknown):
                s = spec("fixture-hosted", 2)
                if unknown:
                    s.input_usd_per_million = None
                e = self.engine([(s, judgment())], max_cost_usd=0)
                r = e.evaluate(self.request(decision_id=str(unknown), stakes="moderate"))
                self.assertEqual(r["usage"]["provider_calls"], 0)
                self.assertIn("BUDGET:unknown_price" if unknown else "BUDGET:cost_limit", r["route"])
                self.assertFalse(self.adapters["fixture-hosted"].calls)

    def test_unknown_usage_is_counted_as_unknown_not_free(self):
        e = self.engine([(spec("fixture-local", 1, local=True),
                          Judgment("BUY", .95))])
        r = e.evaluate(self.request())
        self.assertEqual(r["usage"]["unknown_usage_calls"], 1)
        self.assertEqual(r["usage"]["unknown_cost_calls"], 1)
        metrics = self.ledger.metrics("BUILDER", "business")
        self.assertEqual(metrics["unknown_usage_calls"], 1)
        self.assertEqual(metrics["unknown_cost_calls"], 1)
        self.assertEqual(metrics["completed_without_reasoning"], 1)
        self.assertIn("counterfactual", metrics["savings_basis"])

    def test_idempotency_replays_without_model_call_and_conflicts(self):
        e = self.engine([(spec("fixture-local", 1, local=True), judgment())])
        first = e.evaluate(self.request())
        again = e.evaluate(self.request())
        self.assertTrue(again["replayed"])
        self.assertEqual(again["decision"], first["decision"])
        self.assertEqual(len(self.adapters["fixture-local"].calls), 1)
        self.assertEqual(self.ledger.metrics("BUILDER", "business")["decisions"], 1)
        with self.assertRaisesRegex(DecisionError, "DECISION_ID_CONFLICT"):
            e.evaluate(self.request(state={"signal": 8}))

    def test_scoped_ledger_outcome_calibration_and_private_files(self):
        e = self.engine([(spec("fixture-local", 1, local=True), judgment())])
        for capacity, agent in [("business", "BUILDER"), ("personal", "BUILDER"), ("business", "ANALYST")]:
            e.evaluate(self.request(capacity=capacity, requesting_agent=agent))
        self.ledger.attach_outcome("fixture-decision", "BUILDER", "business",
                                   {"correct_decision": "BUY"}, "simulated action")
        record = self.ledger.inspect("fixture-decision", "BUILDER", "business")
        self.assertEqual(record["outcome"], {"correct_decision": "BUY"})
        self.assertEqual(record["action_taken"], "simulated action")
        for capacity, agent in [("personal", "BUILDER"), ("business", "ANALYST")]:
            self.assertIsNone(self.ledger.inspect("fixture-decision", agent, capacity)["outcome"])
        metrics = self.ledger.metrics("BUILDER", "business")
        self.assertEqual(metrics["calibration"]["fixture-local:general"]["correct"], 1)
        self.assertAlmostEqual(metrics["calibration"]["fixture-local:general"]["brier_binary_proxy"], .0025)
        with self.assertRaisesRegex(DecisionError, "OUTCOME_ALREADY_ATTACHED"):
            self.ledger.attach_outcome("fixture-decision", "BUILDER", "business",
                                       {"correct_decision": "SELL"})
        for file in Path(self.tmp.name).rglob("*.sqlite3"):
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)

    def test_invalid_nan_provider_output_abstains(self):
        e = self.engine([(spec("fixture-local", 1, local=True),
                          Judgment("BUY", float("nan")))])
        r = e.evaluate(self.request())
        self.assertTrue(r["abstain"])
        self.assertEqual(r["provider_results"][0]["reason"], "PROVIDER_ERROR")
        self.assertNotIn("NaN", json.dumps(r))
        with self.assertRaisesRegex(DecisionError, "INVALID_JSON"):
            self.request(state={"signal": float("nan")}).validate()

    def test_invalid_choice_and_usage_rejected(self):
        req = self.request().validate()
        for value in [Judgment("unknown", .9), Judgment("BUY", .9, input_tokens=-1),
                      Judgment("BUY", .9, cost_usd=float("inf"))]:
            with self.subTest(value=value):
                with self.assertRaises(DecisionError):
                    value.validate(req)

    def test_probability_is_decisiveness_not_answer_confidence(self):
        req = self.request(primitive="BOOLEAN_PROBABILITY", options=None).validate()
        self.assertIn("probability", answer_shape(req))
        for probability, expected in [(.5, 0), (.9, .8), (.1, .8)]:
            with self.subTest(probability=probability):
                value = normalize(req, {"probability": probability})
                self.assertEqual(value.decision, probability)
                self.assertAlmostEqual(value.confidence, expected)
                self.assertEqual(value.confidence_kind, "probability_decisiveness")

    def test_rank_tie_abstains_without_false_ordering_certainty(self):
        req = self.request(primitive="RANK").validate()
        value = normalize(req, {"scores": {"BUY": 2, "SELL": 2, "HOLD": 1}, "confidence": .95})
        self.assertTrue(value.abstain)
        self.assertEqual(value.reason, "RANK_TIE")
        self.assertEqual(set(value.decision), set(OPTIONS))

    def test_normalize_rejects_malformed_model_answers(self):
        req = self.request().validate()
        for answer in [{}, {"choice": "BUY"}, {"choice": "NOPE", "confidence": .9},
                       {"choice": "BUY", "confidence": 7}, [], "BUY"]:
            with self.subTest(answer=answer):
                with self.assertRaises(DecisionError):
                    normalize(req, answer)

    def test_model_abstention_is_preserved(self):
        req = self.request().validate()
        value = normalize(req, {"abstain": True, "reason": "NEEDS_MORE_EVIDENCE"})
        self.assertTrue(value.abstain)
        self.assertEqual(value.reason, "NEEDS_MORE_EVIDENCE")

    def test_prompt_marks_state_as_untrusted_and_lists_options(self):
        req = self.request().validate()
        system, user = build_prompt(req, {"state": {"signal": 7}})
        self.assertIn("untrusted", system)
        self.assertIn("BUY|SELL|HOLD", user)

    def test_http_failure_is_single_attempt_and_closes_connection(self):
        with patch("arcturion_decision.providers.http.client.HTTPConnection") as factory:
            conn = factory.return_value
            conn.getresponse.return_value.status = 503
            with self.assertRaisesRegex(DecisionError, "PROVIDER_HTTP_ERROR"):
                post_json("http://fixture.invalid/simulated", {}, 1)
            self.assertEqual(conn.request.call_count, 1)
            conn.close.assert_called_once()

    def test_post_json_rejects_non_http_urls(self):
        for url in ["file:///etc/passwd", "ftp://fixture.invalid/x", "not a url"]:
            with self.subTest(url=url), self.assertRaisesRegex(DecisionError, "INVALID_PROVIDER_URL"):
                post_json(url, {}, 1)


if __name__ == "__main__":
    unittest.main()
