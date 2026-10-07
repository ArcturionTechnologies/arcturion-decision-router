# Consequential consultation

The sequence for a decision that matters is: gather evidence, consult the router, weigh
any disagreement, record an evidence-based disposition, recommend with the matching
receipt, obtain any independently required approval, then act. The router only supplies the
middle steps. It never acts and never approves.

## Which decisions count

The router does not guess risk from prose. The calling workflow declares it:

| Trigger | Declare |
| --- | --- |
| New recurring spend, contract, or a material allocation of time or money | `resource_commitment` |
| A direction or priority that is hard to reverse or crowds out other work | `strategic_tradeoff` |
| Deploy, cutover, migration, production configuration | `production_change` |
| Permission, trust-boundary or exposure changes | `security_change` |
| Sending, publishing, paying or promising something externally | `external_commitment` |
| One-way deletion or other destructive change | `irreversible_action` |

Put these in `workflow.consequences`. Decision types such as `financial`, `legal`,
`medical`, `security`, `payment` and `destructive` also raise stakes to high. Explicit
consequences cannot be lowered by labelling the work arithmetic. Arithmetic, formatting,
exact lookups and executing an already-settled choice are exempt (`workflow.work_kind`).
Do not consult a model to decide whether to consult a model.

## Run it

```sh
python3 -m arcturion_decision decide --input request.json --state-dir ./decision-state
```

A request needs an explicit `requesting_agent` and `capacity` (`business` or `personal`;
two isolation scopes with separate ledgers). `--input-json '...'` accepts short sanitized
JSON, but arguments show up in process lists, so use a file or stdin for anything private.
Requests are limited to 262,144 bytes, and secret-shaped values and keys such as `password`
or `api_key` are rejected.

## Record a review, then verify it

After advice, repeat the **exact** original request inside a review envelope:

```json
{
  "request": { "...": "the exact request that was decided" },
  "review": {
    "reviewer": "BUILDER",
    "disposition": "accept",
    "recommendation": "service",
    "rationale": "The failing health check blocks other work, so it goes first.",
    "evidence_fields": ["facts"],
    "advice_assessment": "The advice agrees with the stated facts and the no-production-change limit."
  }
}
```

Run `review --input envelope.json`. Later, `verify` with
`{"request": {...}, "require_review": true}` checks it with no provider call. Matching covers
agent, capacity, question, all state, options, constraints, workflow metadata and declared risk.
Changed evidence needs a new decision ID.

Dispositions: `accept` (adopt the advice, explain why), `reject` (choose another option,
explain why), `deliberated` (the responsible agent assessed a hand-off; this does not claim
independent corroboration or human approval), `provisional` (consultation was unavailable or
evidence is unresolved; say so). Reviews are immutable. The checks confirm a rationale and
evidence references exist; whether the reasoning is sound is for the reviewer to judge.

## High stakes and failure

High stakes require two independent judgments. Without them the result is an abstention with a
`handoff` for the responsible agent. If every provider is unavailable the record says so, no
paid retry happens, and only provisional analysis is allowed. Every receipt carries
`authorized: false` and `advisory_only: true`.
