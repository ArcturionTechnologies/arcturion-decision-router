"""Arcturion decision router: a bounded, advisory decision cascade with receipts."""
from .protocol import Request, Judgment, Primitive, ProviderSpec, DecisionError
from .engine import Engine
from .ledger import Ledger
from pathlib import Path
import json
import os

__version__ = "0.1.0"


def default_state_root():
    """Ledger folder. Override with ARCTURION_DECISION_STATE or the --state-dir flag."""
    configured = os.environ.get("ARCTURION_DECISION_STATE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "arcturion-decision"


def load_registry(config_path=None):
    """Read a registry file (the --config flag, ARCTURION_DECISION_CONFIG, or the bundled default)."""
    config_path = config_path or os.environ.get("ARCTURION_DECISION_CONFIG")
    path = Path(config_path) if config_path else Path(__file__).with_name("registry.json")
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise DecisionError("INVALID_REGISTRY")
    return config


def create_engine(config_path=None, state_root=None, *, policy=None, adapters=None):
    """Build an Engine from a registry.

    `adapters` maps a provider name to a ready callable and takes precedence over
    the registry's own `adapter` setting. That is the hook for embedding your own judge.
    """
    from .providers import build_adapter
    config = load_registry(config_path)
    providers = [ProviderSpec(**item) for item in config["providers"]]
    if len({p.name for p in providers}) != len(providers):
        raise DecisionError("DUPLICATE_PROVIDER")
    ready = dict(adapters or {})
    for p in providers:
        if p.enabled and p.name not in ready:
            ready[p.name] = build_adapter(p)
    ready = {name: fn for name, fn in ready.items()
             if any(p.name == name and p.enabled for p in providers)}
    return Engine(providers, ready, Ledger(state_root or default_state_root()), policy=policy,
                  coworker_providers=config.get("coworker_providers", []), **config["limits"])
