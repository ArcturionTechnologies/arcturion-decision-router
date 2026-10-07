"""Workflow receipt checks with temporary ledgers and fixture judges. No network, no sends."""
from dataclasses import asdict
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcturion_decision import Engine, Ledger, Request, ProviderSpec, Judgment, DecisionError
from arcturion_decision.consultation import record_review
from arcturion_decision import workflows as wf


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger = Ledger(self.root / "ledger")
        self.calls = []

        def adapter(*args):
            self.calls.append(args)
            return Judgment("stage", .99, receipt_id="fixture-receipt")
        self.engine = Engine([ProviderSpec("fixture", 2, "fixture", ["CHOICE"], 12000, True)],
                             {"fixture": adapter}, self.ledger)

    def recorded(self, payload, *, agent="BUILDER", capacity="business", provisional=False):
        request = Request(
            state={"workflow_binding": wf.workflow_binding(payload), "evidence": "rollback rehearsed"},
            question="Should the reviewed change proceed to approval?",
            options={"stage": "Stage for approval", "defer": "Defer"},
            requesting_agent=agent, capacity=capacity, decision_id="workflow-case", stakes="high",
            decision_type="production_change")
        if provisional:
            def unavailable(*args):
                self.calls.append(args)
                raise RuntimeError("fixture outage")
            self.engine.adapters["fixture"] = unavailable
        self.engine.evaluate(request)
        record_review(self.ledger, request, {
            "reviewer": agent, "disposition": "provisional" if provisional else "deliberated",
            "recommendation": "stage",
            "rationale": "Rollback was rehearsed and current evidence supports staging.",
            "evidence_fields": ["evidence", "workflow_binding"],
            "advice_assessment": "The single available advisor supports staging; independent corroboration is unavailable."})
        return asdict(request)

    def payload(self):
        return wf.approval_payload("BUILDER", "business", "Review staged change", "Test evidence ready",
                                   "owner-decision", ["stage command"])

    def manifest(self):
        return {"schema_version": 1, "files": [{"target": "a", "sha256": "1"}], "decision_required": True,
                "requesting_agent": "BUILDER", "capacity": "business"}

    # approvals

    def test_exact_review_matches_without_provider_call(self):
        p = self.payload()
        request = self.recorded(p)
        count = len(self.calls)
        result = wf.approval_consultation(p, request, ledger=self.ledger)
        self.assertEqual(result["status"], "verified")
        self.assertFalse(result["authorized"])
        self.assertFalse(result["proof"]["review"]["independent_corroboration"])
        self.assertEqual(len(self.calls), count)

    def test_changed_gate_fields_and_wrong_agent_capacity_fail(self):
        p = self.payload()
        request = self.recorded(p)
        for key, value in {"title": "changed", "context": "changed", "gate_class": "spend",
                           "proposed_commands": ["other"], "agent": "ANALYST", "capacity": "personal"}.items():
            changed = dict(p, **{key: value})
            self.assertEqual(wf.approval_consultation(changed, request, ledger=self.ledger)["status"], "invalid", key)

    def test_missing_and_unavailable_remain_explicit_provisional(self):
        p = self.payload()
        self.assertEqual(wf.approval_consultation(p)["status"], "missing")
        request = self.recorded(p, provisional=True)
        result = wf.approval_consultation(p, request, ledger=self.ledger)
        self.assertEqual(result["status"], "provisional")
        self.assertIn("provisional", wf.consultation_notice(result))

    def test_routine_approval_has_no_consultation(self):
        p = dict(self.payload(), gate_class="reversible-in-lane")
        with patch.object(wf, "_verify", side_effect=AssertionError("must not verify routine work")):
            self.assertIsNone(wf.approval_consultation(p))

    def test_host_can_supply_its_own_consequential_classes(self):
        p = dict(self.payload(), gate_class="vendor-onboarding")
        self.assertIsNone(wf.approval_consultation(p))
        self.assertEqual(wf.approval_consultation(p, classes={"vendor-onboarding"})["status"], "missing")

    def test_consequence_metadata_cannot_be_downgraded_or_omitted(self):
        m = self.manifest()
        raw = self.recorded(m)
        for changes in ({"decision_type": "general", "stakes": "low"}, {"decision_type": "general", "stakes": "high"},
                        {"decision_type": "resource_commitment", "stakes": "high"}):
            m["decision_request"] = dict(raw, **changes)
            with self.assertRaisesRegex(DecisionError, "WORKFLOW_CONSEQUENCE_REQUIRED"):
                wf.deployment_consultation(m, ledger=self.ledger)
        p = self.payload()
        p["gate_class"] = "spend"
        raw["state"]["workflow_binding"] = wf.workflow_binding(p)
        self.assertEqual(wf.approval_consultation(p, raw, ledger=self.ledger)["reason"], "WORKFLOW_CONSEQUENCE_REQUIRED")

    def test_broken_consultation_database_does_not_block_human_escalation(self):
        p = self.payload()
        raw = self.recorded(p)
        with patch.object(wf, "verify_receipt", side_effect=sqlite3.DatabaseError("fixture corruption")):
            result = wf.approval_consultation(p, raw, ledger=self.ledger)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["recommendation_status"], "provisional")
        self.assertFalse(result["authorized"])

    # deployments

    def test_new_deployment_missing_receipt_fails(self):
        with self.assertRaisesRegex(DecisionError, "DECISION_REQUEST_REQUIRED"):
            wf.deployment_consultation(self.manifest(), ledger=self.ledger)

    def test_changed_deployment_payload_fails(self):
        m = self.manifest()
        m["decision_request"] = self.recorded(m)
        m["new_metadata"] = "changed evidence"
        with self.assertRaisesRegex(DecisionError, "WORKFLOW_BINDING_MISMATCH"):
            wf.deployment_consultation(m, ledger=self.ledger)

    def test_deliberated_deploy_receipt_remains_advisory(self):
        m = self.manifest()
        m["decision_request"] = self.recorded(m)
        count = len(self.calls)
        proof = wf.deployment_consultation(m, ledger=self.ledger)
        self.assertFalse(proof["authorized"])
        self.assertFalse(proof["review"]["independent_corroboration"])
        self.assertEqual(len(self.calls), count)

    def test_provisional_cannot_deploy(self):
        m = self.manifest()
        m["decision_request"] = self.recorded(m, provisional=True)
        with self.assertRaisesRegex(DecisionError, "PROVISIONAL_ANALYSIS_ONLY"):
            wf.deployment_consultation(m, ledger=self.ledger)

    def test_settled_manifest_is_not_gated(self):
        for manifest in ({"files": []}, {"files": [], "decision_required": False}):
            self.assertIsNone(wf.deployment_consultation(manifest, ledger=self.ledger))
        with self.assertRaisesRegex(DecisionError, "INVALID_DECISION_REQUIRED"):
            wf.deployment_consultation({"decision_required": "yes"}, ledger=self.ledger)

    def test_decision_bearing_manifest_needs_explicit_scope(self):
        with self.assertRaisesRegex(DecisionError, "EXPLICIT_WORKFLOW_SCOPE_REQUIRED"):
            wf.deployment_consultation({"decision_required": True}, ledger=self.ledger)


if __name__ == "__main__":
    unittest.main()
