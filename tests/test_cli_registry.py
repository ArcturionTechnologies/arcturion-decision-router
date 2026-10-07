"""Command line, registry loading and the bundled examples. Temp dirs only, no network."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arcturion_decision import (DecisionError, Judgment, create_engine, default_state_root, load_registry)
from arcturion_decision.cli import main, read_input


def run_cli(*argv, stdin=None):
    out = io.StringIO()
    with patch.object(sys, "argv", ["arcturion-decision", *argv]), contextlib.redirect_stdout(out):
        if stdin is not None:
            with patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin.encode()))):
                code = main()
        else:
            code = main()
    return code, out.getvalue()


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / "state"

    def test_bundled_registry_is_deterministic_only(self):
        config = load_registry()
        self.assertEqual(config["providers"], [])
        engine = create_engine(state_root=self.state)
        self.assertEqual(engine.providers, [])
        self.assertEqual(engine.adapters, {})

    def test_state_root_comes_from_env_not_a_baked_in_path(self):
        with patch.dict(os.environ, {"ARCTURION_DECISION_STATE": self.tmp.name}):
            self.assertEqual(default_state_root(), Path(self.tmp.name))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARCTURION_DECISION_STATE", None)
            self.assertEqual(default_state_root().name, "arcturion-decision")

    def test_invalid_and_duplicate_registries_are_rejected(self):
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"schema_version": 2, "providers": []}))
        with self.assertRaisesRegex(DecisionError, "INVALID_REGISTRY"):
            create_engine(bad, self.state)
        entry = {"name": "x", "tier": 2, "model": "m", "primitives": ["CHOICE"], "context_bytes": 1, "local": True,
                 "enabled": False}
        dup = Path(self.tmp.name) / "dup.json"
        dup.write_text(json.dumps({"schema_version": 1, "providers": [entry, entry], "limits": {}}))
        with self.assertRaisesRegex(DecisionError, "DUPLICATE_PROVIDER"):
            create_engine(dup, self.state)

    def test_hosted_example_loads_and_abstains_cleanly_without_credentials(self):
        env = {"HOSTED_JUDGE_MODEL": "example-model", "LOCAL_JUDGE_MODEL": "example-local"}
        with patch.dict(os.environ, env):
            os.environ.pop("OPENAI_API_KEY", None)
            engine = create_engine(ROOT / "examples/registry.hosted.json", self.state)
            self.assertEqual(sorted(engine.adapters), ["hosted", "local"])  # reasoning is disabled
            hosted = engine.adapters["hosted"]
            from arcturion_decision import Request
            request = Request({"x": 1}, "Q?", {"a": "A", "b": "B"}, requesting_agent="BUILDER",
                              capacity="business").validate()
            result = hosted(request, {"state": {"x": 1}}, 30)
        self.assertTrue(result.abstain)
        self.assertEqual(result.reason, "CREDENTIAL_UNAVAILABLE")

    def test_custom_adapter_factory_runs_through_the_engine(self):
        engine = create_engine(ROOT / "examples/registry.custom.json", self.state)
        request = json.loads((ROOT / "examples/choice.json").read_text())
        request.update(stakes="low", required_confidence=0.3, decision_id="custom-adapter-1")
        from arcturion_decision import Request
        result = engine.evaluate(Request(**request))
        self.assertEqual(result["decision"], "service")
        self.assertEqual(result["models_used"], ["keywords"])
        self.assertFalse(result["authorized"])

    def test_injected_adapter_overrides_registry(self):
        entry = {"name": "fixture", "tier": 2, "model": "m", "primitives": ["CHOICE"], "context_bytes": 12000,
                 "local": True, "enabled": True, "input_usd_per_million": 0, "output_usd_per_million": 0}
        path = Path(self.tmp.name) / "inject.json"
        path.write_text(json.dumps({"schema_version": 1, "providers": [entry], "limits": {}}))
        seen = []
        engine = create_engine(path, self.state, adapters={"fixture": lambda *a: seen.append(a) or Judgment("a", .99)})
        self.assertEqual(list(engine.adapters), ["fixture"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = str(Path(self.tmp.name) / "state")

    def test_deterministic_example_needs_no_model(self):
        code, out = run_cli("evaluate", "--input", str(ROOT / "examples/deterministic.json"),
                            "--state-dir", self.state)
        result = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(result["decision"], 1)
        self.assertEqual(result["route"], ["T0:deterministic"])
        self.assertEqual(result["usage"]["provider_calls"], 0)
        self.assertFalse(result["authorized"])

    def test_inspect_and_metrics_round_trip(self):
        run_cli("gate", "--input", str(ROOT / "examples/deterministic.json"), "--state-dir", self.state)
        code, out = run_cli("inspect", "--agent", "BUILDER", "--capacity", "business",
                            "--decision-id", "example-queue-limit-1", "--state-dir", self.state)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["result"]["decision"], 1)
        code, out = run_cli("metrics", "--agent", "BUILDER", "--capacity", "business", "--state-dir", self.state)
        self.assertEqual(json.loads(out)["decisions"], 1)

    def test_choice_without_providers_hands_off_instead_of_guessing(self):
        code, out = run_cli("decide", "--input", str(ROOT / "examples/choice.json"), "--state-dir", self.state)
        result = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(result["abstain"])
        self.assertIsNone(result["decision"])
        self.assertEqual(result["handoff"]["kind"], "responsible_agent_deliberation")

    def test_stdin_and_human_output(self):
        payload = (ROOT / "examples/deterministic.json").read_text()
        code, out = run_cli("evaluate", "--human", "--state-dir", self.state, stdin=payload)
        self.assertEqual(code, 0)
        self.assertIn("ARCTURION DECISION", out)
        self.assertIn("authorization is always false", out)

    def test_errors_are_reason_codes_not_tracebacks(self):
        code, out = run_cli("decide", "--input-json", '{"state":{"a":1}}', "--state-dir", self.state)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), {"status": "error", "reason": "EXPLICIT_SCOPE_REQUIRED"})
        code, out = run_cli("decide", "--input-json", '{"password":"x"}', "--state-dir", self.state)
        self.assertEqual(json.loads(out)["reason"], "SENSITIVE_INPUT")

    def test_providers_lists_registry(self):
        code, out = run_cli("providers", "--state-dir", self.state)
        self.assertEqual((code, json.loads(out)), (0, {"providers": []}))

    def test_read_input_rejects_non_objects(self):
        with self.assertRaisesRegex(DecisionError, "INVALID_JSON"):
            read_input(direct="[1]")


if __name__ == "__main__":
    unittest.main()
