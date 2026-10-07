# Judge adapters

The router never needs a model. With the bundled registry (`providers: []`) it applies
host policy, required-field checks, context limits and **deterministic rules**
(`threshold` and `argmax`), and for everything else it abstains and returns a minimized
hand-off packet for a person or an agent to decide. Model judges are optional add-ons.

## The interface

An adapter is any callable:

```python
def adapter(request, packet, timeout_seconds) -> Judgment
```

- `request` is the validated `Request` (question, options, primitive, stakes, ids).
- `packet` is the minimized evidence the engine chose to share. It is untrusted data.
- Return a `Judgment`. If you cannot answer, return
  `Judgment(abstain=True, reason="SHORT_CODE", provider_called=False)`.
  Raising is allowed; the engine records `PROVIDER_ERROR` and never exposes the message.

Fill in `input_tokens`, `output_tokens` and `cost_usd` when you know them. Unknown
usage is counted as unknown, never as free.

## Reference adapters (environment variables only)

| Registry `adapter` | Wire | Default key variable | Default base URL |
| --- | --- | --- | --- |
| `openai_compatible` | `POST {base}/chat/completions` | `OPENAI_API_KEY` | `OPENAI_BASE_URL`, else `https://api.openai.com/v1` |
| `anthropic` | `POST {base}/v1/messages` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL`, else `https://api.anthropic.com` |

The registry stores the **names** of variables, never values. `options` accepts
`api_key_env`, `base_url_env`, `base_url`, `api_key_required`, `max_output_tokens`,
`max_call_seconds` and (OpenAI wire only) `json_mode`. Anything else is rejected, so an
inline `api_key` in a registry file fails loudly instead of being committed by accident.
A `model` written as `env:NAME` is read from that variable.

Local servers that speak the Chat Completions wire (llama.cpp, Ollama's OpenAI endpoint,
vLLM) work with `"api_key_required": false` and a loopback `base_url`.

If the key is missing, the adapter abstains with `CREDENTIAL_UNAVAILABLE` and makes no call.

## Writing your own

Point `adapter` at `package.module:factory`. The factory receives the `ProviderSpec` and
returns the callable. See `examples/keyword_adapter.py`, a ~25 line example that uses no model.
This imports code named in your registry, so only use registries you wrote.
You can also skip the registry and pass `adapters={"name": callable}` to `create_engine`.

## What the engine does around any adapter

- Picks tiers in order (rules, local, primary, second judge, reasoning), at most `max_calls` per decision.
- Reserves a worst-case cost before each paid call and stops at `max_cost_usd`.
  A paid provider with no configured price is not called.
- Never retries automatically, never fans out to every provider.
- Treats providers in the same `independent_group` as one source, not as corroboration.
- Lets strong disagreement (both confidences at least 0.65) veto a majority.
- Marks every result `authorized: false`. Advice is not permission to act.
