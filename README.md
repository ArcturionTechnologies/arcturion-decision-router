# ArcturionDecisionRouter

A small decision router for AI agents. It answers bounded questions such as "which of these
three tasks first?" with rules where rules are enough, optional model judges where they are not,
hard limits on cost and calls, and a private receipt for every answer.

Most agent setups either let a model decide everything or leave every choice to a person. This
sits in between. A rule settles what a rule can settle. A model is asked only when needed, only
within a budget, and its advice is checked, recorded and never treated as permission. When the
evidence is thin or the judges disagree, it says so and hands the question back with a minimal
packet instead of guessing.

Python 3.11+, standard library only. **No model or network is required to run it.**

> **Portfolio project.** This is an open-source sample of the tooling behind Arcturion's
> multi-agent setup. It is not a commercial product and makes no claims about revenue or
> customers. All bundled examples are synthetic.

## Quickstart

```bash
git clone https://github.com/ArcturionTechnologies/arcturion-decision-router.git
cd arcturion-decision-router

# A rule decides. Zero model calls, and the result is saved to a private ledger.
python3 -m arcturion_decision evaluate \
  --input examples/deterministic.json --state-dir ./decision-state
# {"decision":1.0, "reason":"DETERMINISTIC_RULE", "route":["T0:deterministic"],
#  "usage":{"provider_calls":0,...}, "authorized":false, ...}

# A judgment call with no judges configured: it abstains and hands off, it does not guess.
python3 -m arcturion_decision decide --human \
  --input examples/choice.json --state-dir ./decision-state
```

Or install it as a command: `pip install .`, then run `arcturion-decision ...`.

## How a decision flows

```
request (explicit agent + scope, compact state)
  -> validation, secret screen, host policy, declared-risk classification
  -> context projection (never silently truncated)
  -> tier 0  deterministic rules (threshold, argmax)
  -> tier 1  local judge          -> tier 2 primary judge
  -> at most one more independent judge when confidence is low or stakes are high
  -> tier 4  reasoning judge, or an explicit hand-off
  -> private SQLite ledger, with an optional outcome recorded later
```

Things it does on purpose:

- **Rules first.** `threshold` and `argmax` rules in the request run before any model.
- **Bounded.** Default limits are 3 calls, 60 seconds and $0.05, reserved before each paid call.
  A paid judge with no configured price is not called. There are no automatic retries.
- **Independent corroboration.** Judges in the same `independent_group` count as one source.
  Confidences are weighted, and strong disagreement vetoes a majority. High stakes need two
  independent judgments or the result is a hand-off.
- **Advice, not authority.** Every result carries `authorized: false`.
- **Idempotent.** A decision ID is reserved before any call, so a retry replays the stored result
  and cannot trigger a second paid call. The same ID with different content is rejected.
- **Honest accounting.** Unknown token counts and costs are counted as unknown, never as free.
- **Private by default.** Ledgers are per scope and agent, folders mode 0700, files 0600. Raw
  state is not stored, only hashes and normalized results. Secret-shaped input is rejected.

Primitives: `CHOICE`, `SCORE`, `BOOLEAN_PROBABILITY`, `RANK`, `UTILITY`, `RISK`. Rank ties abstain
instead of inventing an order.

## Model judges are optional adapters

A judge is any function `adapter(request, packet, timeout) -> Judgment`. Two reference adapters
ship, driven entirely by environment variables whose *names* you put in a registry file:

```bash
export OPENAI_API_KEY=...            # your own key, from your own environment
export HOSTED_JUDGE_MODEL=your-model-name
python3 -m arcturion_decision decide --config examples/registry.hosted.json \
  --input examples/choice.json --state-dir ./decision-state
```

`openai_compatible` also covers local servers that speak the same wire, and `anthropic` covers the
Messages API. You can write your own in a few lines (`examples/keyword_adapter.py`). With no key
set, an adapter abstains with `CREDENTIAL_UNAVAILABLE` and makes no call. See
[docs/ADAPTERS.md](docs/ADAPTERS.md). Keys are never read from files in this project.

## Python

```python
from arcturion_decision import create_engine

engine = create_engine(state_root="./decision-state")   # bundled registry: rules only
result = engine.decide(
    state={"queue_depth": 4}, question="Is the queue within the limit?",
    primitive="BOOLEAN_PROBABILITY", requesting_agent="builder", capacity="business",
    decision_id="queue-check-1",
    constraints={"deterministic": {"op": "threshold", "field": "queue_depth",
                                   "operator": "lte", "value": 10}})
```

Also: `engine.score`, `engine.rank`, `engine.gate`, `engine.evaluate(Request(...))`, and a host
`policy(request)` callback that can block any request before models run.

## Consequential decisions and receipts

For decisions that commit money, change production or security, or can't be undone, the calling
workflow declares the consequence, consults, records an evidence-based review, and later *verifies*
the receipt without another call. See [docs/CONSULTATION.md](docs/CONSULTATION.md) and
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Limits, stated plainly

- Default confidence thresholds (0.80 low, 0.85 moderate, 0.90 high) are engineering defaults, not
  validated error rates. Self-reported model confidence is not calibration.
- Risk is taken from declared metadata, not inferred from prose.
- The ledger is a local file protected by filesystem permissions. This is not a multi-tenant server.
- Secret screening catches common credential shapes. It is not a personal-data sanitizer.

## Project layout

```
arcturion_decision/protocol.py     request, judgment, validation, secret screen
arcturion_decision/engine.py       the bounded cascade, stability and veto rules
arcturion_decision/providers.py    adapter interface and the two reference adapters
arcturion_decision/ledger.py       private SQLite ledger, outcomes, metrics
arcturion_decision/consultation.py classification, review, receipt verification
arcturion_decision/workflows.py    approval and deployment receipt checks
arcturion_decision/cli.py          command line
examples/                          synthetic requests and registries
tests/                             85 tests, stdlib unittest
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Everything runs in temporary folders with fixture judges: no network, no credentials.

## License

MIT. See [LICENSE](LICENSE).

Implementation is AI-assisted; architecture, requirements, and testing directed by Robert Lingoes.
