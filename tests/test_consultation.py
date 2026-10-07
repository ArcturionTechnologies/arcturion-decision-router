"""Consequential receipt controls: scope, evidence changes, failure, exclusions, input."""
from pathlib import Path
from dataclasses import asdict
import copy, importlib.util, json, sys, tempfile, unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcturion_decision import Engine, Ledger, Request, ProviderSpec, Judgment, DecisionError
from arcturion_decision.protocol import digest
from arcturion_decision.consultation import classify, record_review, verify_receipt, check_workflow
from arcturion_decision.cli import read_input

class ConsultationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.ledger=Ledger(Path(self.tmp.name)/"state")
        self.calls=[]
        spec=ProviderSpec("fixture",2,"fixture",["CHOICE"],12000,True)
        def adapter(*args):
            self.calls.append(args)
            return Judgment("guide",.99,receipt_id="fixture-receipt")
        self.engine=Engine([spec],{"fixture":adapter},self.ledger)
    def request(self,**overrides):
        payload=dict(state={"deadline":"expert leaves today","impact":"guide reduces repeated work"},
                     question="Which work block?",options={"guide":"Write guide","labels":"Change labels"},
                     requesting_agent="BUILDER",capacity="business",decision_id="bounded-priority",stakes="moderate")
        payload.update(overrides);return Request(**payload)
    def review(self,**overrides):
        value=dict(reviewer="BUILDER",disposition="accept",recommendation="guide",
                   rationale="The expert leaves today, making the guide time sensitive.",
                   evidence_fields=["deadline"],advice_assessment="Advice agrees with the stated deadline and reduces rework.")
        value.update(overrides);return value
    def test_direct_json_parser_retains_size_and_secret_checks(self):
        self.assertEqual(read_input(direct='{"state":{"safe":"yes"}}'),{"state":{"safe":"yes"}})
        for text,reason in [('[]',"INVALID_JSON"),('x'*262145,"REQUEST_TOO_LARGE"),
                            ('{"password":"fixture"}',"SENSITIVE_INPUT")]:
            with self.subTest(reason=reason),self.assertRaisesRegex(DecisionError,reason):read_input(direct=text)
        path=Path(self.tmp.name)/"input.json";path.write_text('{"safe":true}')
        self.assertEqual(read_input(path),{"safe":True})
    def test_workflow_metadata_consequences_override_exemption(self):
        request=self.request(stakes="low",workflow={"work_kind":"arithmetic"})
        self.assertTrue(classify(request)["exempt"])
        request.workflow["consequences"]=["production_change"]
        self.assertTrue(classify(request)["required"])
        self.assertEqual(classify(request)["stakes"],"high")
        request.workflow["consequences"]=["guessed_prose"]
        with self.assertRaisesRegex(DecisionError,"INVALID_WORKFLOW_CONSEQUENCES"):classify(request)
    def test_exact_review_and_replay_do_not_call_provider(self):
        request=self.request()
        self.engine.evaluate(request)
        proof=record_review(self.ledger,request,self.review())
        self.assertTrue(proof["verified"]);self.assertFalse(proof["authorized"])
        self.assertFalse(proof["review"]["independent_corroboration"])
        self.assertEqual(record_review(self.ledger,request,self.review()),proof)
        self.assertTrue(check_workflow(self.engine,request)["verified"])
        self.engine.evaluate(request);self.assertEqual(len(self.calls),1)
    def test_changed_evidence_question_options_scope_fail(self):
        request=self.request();self.engine.evaluate(request)
        for changes in ({"state":{"deadline":"tomorrow"}},{"question":"Different question?"},
                        {"options":{"guide":"Different","labels":"Labels"}},{"capacity":"personal"},
                        {"requesting_agent":"ANALYST"}):
            with self.subTest(changes=changes),self.assertRaises(DecisionError):
                verify_receipt(self.ledger,self.request(**changes))
        with self.assertRaisesRegex(DecisionError,"DECISION_ID_CONFLICT"):
            self.engine.evaluate(self.request(state={"deadline":"tomorrow"}))
        self.engine.evaluate(self.request(decision_id="new-evidence",state={"deadline":"tomorrow"}))
        self.assertEqual(len(self.calls),2)
    def test_review_required_and_immutable_and_evidence_checked(self):
        request=self.request();self.engine.evaluate(request)
        with self.assertRaisesRegex(DecisionError,"EVIDENCE_BASED_REVIEW_REQUIRED"):check_workflow(self.engine,request)
        for changes in ({"reviewer":"ANALYST"},{"rationale":"agree"},{"evidence_fields":["missing"]},
                        {"disposition":"accept","recommendation":"labels"}):
            with self.subTest(changes=changes),self.assertRaises(DecisionError):
                record_review(self.ledger,request,self.review(**changes))
        record_review(self.ledger,request,self.review())
        with self.assertRaisesRegex(DecisionError,"REVIEW_ALREADY_ATTACHED"):
            record_review(self.ledger,request,self.review(rationale="A newly worded rationale must not overwrite the original review."))
    def test_disagreement_is_explicit_rejection_with_rationale(self):
        request=self.request();self.engine.evaluate(request)
        proof=record_review(self.ledger,request,self.review(disposition="reject",recommendation="labels",
            rationale="Impact evidence says label confusion dominates; the expert deadline alone is insufficient.",
            evidence_fields=["impact"],advice_assessment="Disagree with advice because the stated impact needs prioritizing."))
        self.assertEqual(proof["review"]["disposition"],"reject")
    def test_high_stakes_handoff_keeps_corroboration_and_authority_distinct(self):
        request=self.request(decision_type="production_change",stakes="low")
        result=self.engine.evaluate(request)
        self.assertTrue(result["abstain"]);self.assertEqual(result["escalation_reason"],"INDEPENDENT_JUDGE_UNAVAILABLE")
        self.assertEqual(result["handoff"]["owner"],"BUILDER")
        with self.assertRaisesRegex(DecisionError,"HANDOFF_DELIBERATION"):record_review(self.ledger,request,self.review())
        proof=record_review(self.ledger,request,self.review(disposition="deliberated"))
        self.assertFalse(proof["authorized"]);self.assertFalse(proof["review"]["independent_corroboration"])
    def test_failure_allows_only_provisional_no_paid_retry(self):
        def fail(*args):
            self.calls.append(args);raise RuntimeError("fixture outage")
        self.engine.adapters["fixture"]=fail
        request=self.request()
        result=self.engine.evaluate(request)
        self.assertTrue(result["abstain"]);self.assertFalse(result["handoff"]["automatic_paid_retry"])
        with self.assertRaisesRegex(DecisionError,"UNAVAILABLE_REQUIRES_PROVISIONAL"):
            record_review(self.ledger,request,self.review(disposition="deliberated"))
        proof=record_review(self.ledger,request,self.review(disposition="provisional",
            advice_assessment="Consultation was unavailable; this recommendation is provisional."))
        self.assertEqual(proof["review"]["disposition"],"provisional")
        self.engine.evaluate(request);self.assertEqual(len(self.calls),1)
    def test_exempt_work_needs_no_receipt(self):
        for kind in ("arithmetic","formatting","exact_lookup","settled_execution"):
            request=self.request(stakes="low",workflow={"work_kind":kind})
            self.assertFalse(check_workflow(self.engine,request)["consultation_required"])
        self.assertEqual(self.calls,[])
    def test_non_choice_review_recommendations_are_typed_and_bounded(self):
        for primitive, options, invalid in (
            ("RANK", {"guide":"Guide","labels":"Labels"}, "not-a-rank"),
            ("SCORE", ["Low","High"], 3),
            ("BOOLEAN_PROBABILITY", None, 1.5),
        ):
            with self.subTest(primitive=primitive):
                request=self.request(primitive=primitive, options=options, decision_id=primitive)
                # No configured adapter supports these primitives: real unavailable record.
                self.engine.evaluate(request)
                with self.assertRaisesRegex(DecisionError,"INVALID_RECOMMENDATION"):
                    record_review(self.ledger,request,self.review(disposition="provisional",recommendation=invalid))

    def test_v1_receipt_replay_compatibility_no_changed_metadata(self):
        request=self.request();self.engine.evaluate(request)
        legacy=asdict(request);legacy.pop("workflow")
        with self.ledger.connect("BUILDER","business") as db:
            db.execute("UPDATE decisions SET request_hash=? WHERE decision_id=?",(digest(legacy),request.decision_id))
        self.assertTrue(verify_receipt(self.ledger,request)["verified"])
        self.engine.evaluate(request);self.assertEqual(len(self.calls),1)
        with self.assertRaisesRegex(DecisionError,"RECEIPT_REQUEST_MISMATCH"):
            verify_receipt(self.ledger,self.request(workflow={"decision_required":True}))
if __name__=="__main__":unittest.main()
