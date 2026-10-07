"""Regression tests: policy replay, risk floors, and the reference HTTP adapters (no network)."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcturion_decision import Engine, Ledger, Request, ProviderSpec, Judgment, DecisionError
from arcturion_decision.providers import (AnthropicAdapter, OpenAICompatibleAdapter, build_adapter)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Ledger(Path(self.tmp.name) / "state")

    def test_policy_rechecked_before_replaying_decision(self):
        blocked = [False]
        engine = Engine([], {}, self.ledger, policy=lambda r: blocked[0])
        request = Request({"value": 1}, "Under two?", primitive="BOOLEAN_PROBABILITY", requesting_agent="BUILDER",
                          capacity="business", decision_id="replay-policy",
                          constraints={"deterministic": {"op": "threshold", "field": "value", "operator": "lt", "value": 2}})
        self.assertFalse(engine.evaluate(request)["abstain"])
        blocked[0] = True
        result = engine.evaluate(request)
        self.assertTrue(result["abstain"])
        self.assertIsNone(result["decision"])
        self.assertEqual(result["reason"], "POLICY_BLOCKED")

    def test_consequential_category_cannot_downgrade_stakes(self):
        calls = []
        spec = ProviderSpec("local", 1, "fixture", ["CHOICE"], 12000, True)
        engine = Engine([spec], {"local": lambda *a: calls.append(a) or Judgment("yes", 1)}, self.ledger)
        result = engine.evaluate(Request({"evidence": 1}, "Payment?", {"yes": "Pay", "no": "Wait"},
                                         decision_type="payment", stakes="low", required_confidence=0,
                                         requesting_agent="BUILDER", capacity="business"))
        self.assertEqual(result["risk"], "high")
        self.assertFalse(result["authorized"])
        self.assertTrue(result["abstain"])
        # A single local judge may only corroborate; it is never the routine answer for a high-stakes call.
        self.assertNotEqual(result["reason"], "MODEL_JUDGMENT")

    def test_no_providers_still_hands_off_with_packet(self):
        engine = Engine([], {}, self.ledger)
        result = engine.evaluate(Request({"evidence": "x"}, "Proceed?", {"go": "Go", "wait": "Wait"},
                                         requesting_agent="BUILDER", capacity="business", decision_id="no-providers"))
        self.assertTrue(result["abstain"])
        self.assertEqual(result["escalation_reason"], "PROVIDER_UNAVAILABLE")
        self.assertIn("escalation_packet", result)
        self.assertFalse(result["handoff"]["automatic_paid_retry"])


class AdapterTests(unittest.TestCase):
    def request(self, **kw):
        values = dict(state={"facts": ["A", "B"]}, question="Proceed?", options={"proceed": "Act", "wait": "Wait"},
                      requesting_agent="BUILDER", capacity="business")
        values.update(kw)
        return Request(**values).validate()

    def packet(self, request):
        return {"question": request.question, "state": request.state, "unresolved": "DECISION_UNSTABLE"}

    def openai_reply(self, text, finish="stop", usage=None):
        return {"choices": [{"finish_reason": finish, "message": {"content": text}}],
                "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20}}

    def test_openai_adapter_wire_shape_cost_and_env_only_secret(self):
        seen = {}
        def transport(url, payload, timeout, *, headers=None):
            seen.update(url=url, payload=payload, headers=headers, timeout=timeout)
            return self.openai_reply('{"choice": "wait", "confidence": 0.94}')
        env = {"TEST_KEY": "fixture-key-value", "TEST_BASE": "http://127.0.0.1:9999/v1/"}
        adapter = OpenAICompatibleAdapter("test-model", input_usd_per_million=2, output_usd_per_million=8,
                                          api_key_env="TEST_KEY", base_url_env="TEST_BASE",
                                          transport=transport, environ=env)
        request = self.request()
        result = adapter(request, self.packet(request), 30)
        self.assertEqual(result.decision, "wait")
        self.assertAlmostEqual(result.cost_usd, .00036)
        self.assertEqual((result.input_tokens, result.output_tokens), (100, 20))
        self.assertEqual(seen["url"], "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(seen["headers"], {"Authorization": "Bearer fixture-key-value"})
        self.assertNotIn("tools", seen["payload"])
        self.assertEqual(seen["payload"]["max_tokens"], 512)
        self.assertEqual(seen["payload"]["response_format"], {"type": "json_object"})
        self.assertNotIn("fixture-key-value", json.dumps(seen["payload"]))

    def test_missing_credential_abstains_without_calling(self):
        transport = Mock()
        adapter = OpenAICompatibleAdapter("test-model", api_key_env="TEST_KEY", transport=transport, environ={})
        request = self.request()
        result = adapter(request, self.packet(request), 30)
        self.assertTrue(result.abstain)
        self.assertEqual(result.reason, "CREDENTIAL_UNAVAILABLE")
        self.assertFalse(result.provider_called)
        transport.assert_not_called()

    def test_local_server_without_key_is_allowed_and_free(self):
        seen = {}
        def transport(url, payload, timeout, *, headers=None):
            seen["headers"] = headers
            return self.openai_reply('```json\n{"choice": "proceed", "confidence": 0.9}\n```', usage={})
        adapter = OpenAICompatibleAdapter("local-model", input_usd_per_million=0, output_usd_per_million=0,
                                          base_url="http://127.0.0.1:8080/v1", api_key_required=False,
                                          base_url_env="UNSET_BASE", transport=transport, environ={})
        request = self.request()
        result = adapter(request, self.packet(request), 30)
        self.assertEqual(result.decision, "proceed")
        self.assertEqual(seen["headers"], {})
        self.assertEqual(result.cost_usd, 0.0)

    def test_truncated_or_garbled_output_is_an_error_not_a_decision(self):
        request = self.request()
        for reply in [self.openai_reply('{"choice": "wa', finish="length"), self.openai_reply("no json here"),
                      {"unexpected": True}]:
            with self.subTest(reply=reply):
                adapter = OpenAICompatibleAdapter("m", api_key_env="K", transport=lambda *a, **k: reply,
                                                  environ={"K": "fixture-key-value"})
                with self.assertRaises(DecisionError):
                    adapter(request, self.packet(request), 30)

    def test_engine_contains_adapter_failures(self):
        def broken(*args, **kwargs):
            raise DecisionError("PROVIDER_HTTP_ERROR")
        adapter = OpenAICompatibleAdapter("m", api_key_env="K", transport=broken, environ={"K": "fixture-key-value"})
        spec = ProviderSpec("hosted", 2, "m", ["CHOICE"], 12000, False,
                            input_usd_per_million=1, output_usd_per_million=1, independent_group="hosted")
        engine = Engine([spec], {"hosted": adapter}, Ledger(Path(tempfile.mkdtemp(dir=self.tmpdir()))))
        result = engine.evaluate(Request({"x": 1}, "Q?", {"a": "A", "b": "B"}, stakes="moderate",
                                         requesting_agent="BUILDER", capacity="business", decision_id="broken"))
        self.assertTrue(result["abstain"])
        self.assertEqual(result["provider_results"][0]["reason"], "PROVIDER_ERROR")

    def tmpdir(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return holder.name

    def test_anthropic_adapter_wire_shape(self):
        seen = {}
        def transport(url, payload, timeout, *, headers=None):
            seen.update(url=url, payload=payload, headers=headers)
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"score": 2, "confidence": 0.9}'}],
                    "usage": {"input_tokens": 50, "output_tokens": 10}}
        adapter = AnthropicAdapter("test-model", input_usd_per_million=3, output_usd_per_million=15,
                                   api_key_env="TEST_KEY", transport=transport, environ={"TEST_KEY": "fixture-key-value"})
        request = self.request(primitive="SCORE", options=["Low", "Medium", "High"])
        result = adapter(request, self.packet(request), 30)
        self.assertEqual(result.decision, 2)
        self.assertEqual(seen["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(seen["headers"]["x-api-key"], "fixture-key-value")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(seen["payload"]["messages"][0]["role"], "user")
        self.assertAlmostEqual(result.cost_usd, (50 * 3 + 10 * 15) / 1e6)

    def test_anthropic_max_tokens_stop_is_incomplete(self):
        adapter = AnthropicAdapter("m", api_key_env="K", environ={"K": "fixture-key-value"},
                                   transport=lambda *a, **k: {"stop_reason": "max_tokens", "content": []})
        request = self.request()
        with self.assertRaisesRegex(DecisionError, "PROVIDER_INCOMPLETE"):
            adapter(request, self.packet(request), 30)

    def test_build_adapter_from_registry_entry(self):
        spec = ProviderSpec("hosted", 2, "env:TEST_MODEL_NAME", ["CHOICE"], 12000, False,
                            input_usd_per_million=1, output_usd_per_million=2, adapter="openai_compatible",
                            options={"api_key_env": "MY_KEY", "base_url_env": "MY_BASE", "json_mode": False})
        with self.assertRaisesRegex(DecisionError, "INVALID_ADAPTER_MODEL"):
            build_adapter(spec)  # env:TEST_MODEL_NAME is unset
        with patch.dict("os.environ", {"TEST_MODEL_NAME": "model-from-env"}):
            adapter = build_adapter(spec)
        self.assertEqual(adapter.model, "model-from-env")
        self.assertIsInstance(adapter, OpenAICompatibleAdapter)
        self.assertEqual(adapter.api_key_env, "MY_KEY")
        self.assertFalse(adapter.json_mode)
        self.assertEqual(build_adapter(ProviderSpec("a", 2, "m", ["CHOICE"], 1, False, adapter="anthropic")).__class__,
                         AnthropicAdapter)

    def test_build_adapter_rejects_unknown_or_unconfigured(self):
        for adapter, options, reason in [("", {}, "NOT_CONFIGURED"), ("no_such_adapter", {}, "NOT_INSTALLED"),
                                         ("no_such_module_xyz:make", {}, "NOT_INSTALLED"),
                                         ("openai_compatible", {"api_key": "inline-secret-not-allowed"}, "INVALID_ADAPTER_OPTIONS")]:
            with self.subTest(adapter=adapter):
                spec = ProviderSpec("p", 2, "m", ["CHOICE"], 1, False, adapter=adapter, options=options)
                with self.assertRaisesRegex(DecisionError, reason):
                    build_adapter(spec)

    def test_inline_secrets_are_rejected_in_request_state(self):
        with self.assertRaisesRegex(DecisionError, "SENSITIVE_INPUT"):
            Request({"api_key": "x"}, "Q?", {"a": "A", "b": "B"}, requesting_agent="BUILDER", capacity="business").validate()


if __name__ == "__main__":
    unittest.main()
