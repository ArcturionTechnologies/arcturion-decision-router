"""Supplemental (coworker) providers: an extra opinion that can never replace the primary judge."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcturion_decision import Engine, Ledger, Request, Judgment, ProviderSpec
from arcturion_decision.protocol import DecisionError

def provider(name, tier, context=6000):
    return ProviderSpec(name, tier, "fixture", ["CHOICE"], context, name == "local",
                        input_usd_per_million=0, output_usd_per_million=0, independent_group=name)

class SupplementalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.primary = Mock(return_value=Judgment("repair", .96, cost_usd=0))
        self.local = Mock(return_value=Judgment("repair", .72, cost_usd=0))
        self.engine = Engine([provider("local", 1), provider("primary", 2, 12000)],
                             {"primary": self.primary, "local": self.local}, Ledger(Path(self.tmp.name)),
                             coworker_providers=["local"])
        self.request = Request({"probe": "failed"}, "Which investigation comes first?",
                               {"repair": "Repair service", "labels": "Rename labels"},
                               requesting_agent="BUILDER", capacity="business", decision_id="coworker-test")

    def test_primary_first_and_supplemental_opinion_even_when_primary_confident(self):
        order = []
        self.primary.side_effect = lambda *args: (order.append("primary") or Judgment("repair", .96, cost_usd=0))
        self.local.side_effect = lambda *args: (order.append("local") or Judgment("repair", .72, cost_usd=0))
        r = self.engine.evaluate(self.request)
        self.assertEqual(order, ["primary", "local"])
        self.assertEqual(r["decision"], "repair")
        self.assertEqual(r["coworkers"]["local"]["status"], "opinion_recorded")
        self.assertEqual(r["provider_results"][1]["role"], "coworker")
        self.assertFalse(r["authorized"])

    def test_unavailable_supplement_preserves_primary_result(self):
        self.local.return_value = Judgment(abstain=True, reason="LOCAL_MODEL_BUSY", provider_called=False, cost_usd=0)
        r = self.engine.evaluate(self.request)
        self.assertFalse(r["abstain"])
        self.assertEqual(r["coworkers"]["local"]["reason"], "LOCAL_MODEL_BUSY")

    def test_high_stakes_keeps_independence_gate(self):
        self.request.stakes = "high"
        self.local.return_value = Judgment(abstain=True, reason="LOCAL_RESOURCE_BLOCKED", provider_called=False, cost_usd=0)
        r = self.engine.evaluate(self.request)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["escalation_reason"], "INDEPENDENT_JUDGE_UNAVAILABLE")

    def test_supplement_cannot_replace_unavailable_primary(self):
        self.primary.return_value = Judgment(abstain=True, reason="PRIMARY_UNAVAILABLE", provider_called=False, cost_usd=0)
        self.local.return_value = Judgment("repair", .99, cost_usd=0)
        r = self.engine.evaluate(self.request)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["escalation_reason"], "PRIMARY_JUDGE_UNAVAILABLE")

    def test_strong_disagreement_is_preserved_and_hands_off(self):
        self.local.return_value = Judgment("labels", .90, cost_usd=0)
        r = self.engine.evaluate(self.request)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["reason"], "DECISION_UNSTABLE")
        self.assertEqual([p["decision"] for p in r["provider_results"]], ["repair", "labels"])

    def test_replay_does_not_call_again(self):
        first = self.engine.evaluate(self.request)
        replay = self.engine.evaluate(self.request)
        self.assertEqual(first["provider_results"], replay["provider_results"])
        self.assertEqual(self.local.call_count, 1)
        self.assertEqual(self.primary.call_count, 1)

    def test_deterministic_requests_never_call_providers(self):
        self.request.constraints = {"deterministic": {"op": "argmax", "field": "values"}}
        self.request.state = {"values": {"repair": 2, "labels": 1}}
        r = self.engine.evaluate(self.request)
        self.assertEqual(r["decision"], "repair")
        self.local.assert_not_called()
        self.primary.assert_not_called()

    def test_oversize_supplement_context_is_not_silently_truncated(self):
        self.request.state = {"probe": "x" * 6500}
        r = self.engine.evaluate(self.request)
        self.assertFalse(r["abstain"])
        self.local.assert_not_called()
        self.assertEqual(r["coworkers"]["local"]["status"], "not_called")
