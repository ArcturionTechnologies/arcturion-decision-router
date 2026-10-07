# Workflow receipt checks

`arcturion_decision.workflows` lets an existing approval or deployment flow *verify* that a
matching, reviewed consultation exists. It makes no provider calls, sends nothing and grants no
authority. Your flow keeps its own approval rules.

## Approval requests

```python
from arcturion_decision import workflows as wf

payload = wf.approval_payload("BUILDER", "business", "Review staged change",
                              "Test evidence ready", "owner-decision", ["stage command"])
binding = wf.workflow_binding(payload)   # put this in the decision request: state.workflow_binding
status = wf.approval_consultation(payload, decision_request, ledger=ledger)
print(wf.consultation_notice(status))
```

- Only the classes in `CONSEQUENTIAL_APPROVAL_CLASSES` are checked (pass `classes=` to use your
  own set). Routine classes return `None`.
- The status is `verified`, `missing`, `invalid`, `provisional` or `unavailable`. A missing or
  broken receipt never blocks asking a human; it only labels the recommendation provisional.
- The request must declare high stakes and a matching consequence category (`owner-decision`
  accepts any recognized one). Changing the title, context, class, commands, agent or capacity
  invalidates the binding.

## New deployment judgments

`wf.deployment_consultation(manifest, ledger=...)` returns `None` unless the manifest sets
`decision_required: true`. When it does, the manifest needs `requesting_agent`, `capacity` and
the exact reviewed `decision_request`, which must declare `production_change`. The binding is a
hash of the whole manifest except `decision_request`, so any edit needs a fresh receipt. Missing,
invalid or provisional consultation raises, so call it before touching anything. A `deliberated`
review passes as advisory evidence only. Existing execution approvals still apply.
