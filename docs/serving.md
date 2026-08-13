# Serving NextSearch-1

The three releases are ordinary open-weight causal LMs. Nothing in this
harness requires a special server: it speaks OpenAI-compatible chat
completions with tool calling, so vLLM, SGLang, or any compatible runtime
works.

What does matter is that the serving configuration matches the one the models
were trained and evaluated under. A research policy is a policy over *tool
calls*, so a server that parses tool calls differently, or that drops the
model's thinking between turns, changes behavior in ways that look like a
weaker model rather than a misconfiguration.

| Release | Base | Params | Approx. bf16 weights |
|---|---|---|---|
| NextSearch-1-M | Inkling-Small (MoE) | 276B total, 12B active | ~530 GB |
| NextSearch-1-S | Qwen3.6-35B-A3B (MoE) | 35B total, 3B active | ~72 GB |
| NextSearch-1-XS | Qwen3.5-9B (hybrid attention, dense) | 9B | ~19 GB |

All three are released as merged full weights under Apache 2.0.

## Quick start

```bash
vllm serve NextTokenAI/NextSearch-1-XS \
  --served-model-name NextTokenAI/NextSearch-1-XS \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 65536
```

Then point the harness at it and run one benchmark:

```bash
NEXTSEARCH_BASE_URL=http://localhost:8000/v1 nextsearch-eval run --benches seal0 --models nextsearch-1-xs --n 10
```

## Sampling

These are the settings every reported number used. They are already the
defaults on the registry entries in `nextsearch/models/registry.py`, so you
only need them if you are calling the models from your own code.

| Setting | Value | Why |
|---|---|---|
| `temperature` | 0.7 | As trained. Greedy decoding was not evaluated. |
| `max_tokens` | 16384 | Per turn, not per episode. |
| context | 65536 | See the context note below. |
| thinking | on | These are reasoning models; the research policy was trained with thinking enabled. |

**NextSearch-1-M additionally runs at a continuous reasoning-effort of 0.7**,
which is the value it was trained at. How to pass that through depends on your
server's support for its base model's effort control — check that your served
template actually honors it rather than silently ignoring it, because a
mismatch here changes decode length and therefore both cost and accuracy.

## The context setting is not just a capacity knob

The harness stops an episode gracefully when the conversation reaches
`--max-context`, warns the model at 80% and 90% of that budget, and spends one
final no-tools call to salvage an answer from research already in the
transcript.

All three of those mechanisms only work if the policy cap sits **below** the
model's real serving window. Set the cap at or above the window and every one
of them goes dead at once: overflows stop arriving as graceful stops and start
arriving as provider errors that return nothing, turning finished research
into a zero. The nested subagent default (48k under a 64k window) is sized for
exactly this reason. If you serve at a different context length, move the cap
with it.

## Tool calling

The harness sends standard OpenAI-format tool specs and reads
`message.tool_calls` back. Two things to verify on a new serving setup, both
of which fail quietly:

1. **The tool-call parser matches the model's chat template.** A mismatched
   parser returns an empty `tool_calls` array while the model's intended call
   sits in the text channel as literal markup. The episode then ends with
   `stop_reason=empty_response` and scores zero, which looks exactly like a
   model that refused to answer.
2. **Thinking is returned, not swallowed.** The client normalizes both
   `reasoning_content` and `reasoning`; if your server emits neither, the
   model's thinking is being dropped.

The fastest check is a single episode with tracing on:

```bash
nextsearch-eval run --benches seal0 --models nextsearch-1-xs --n 1
```

A healthy episode reports `stop=final` with two or more turns and at least one
search call. `stop=empty_response` on turn one is the parser problem above.

## Using the harness against your own model

Nothing here is specific to the NextSearch-1 weights. Any OpenAI-compatible
endpoint works:

```bash
nextsearch-eval run --benches main \
  --models my-model --base-url http://localhost:8000/v1
```

and any model served by OpenRouter works by passing its id directly:

```bash
nextsearch-eval run --benches seal0 --models deepseek/deepseek-v4-flash --n 20
```

## Search backend

The evaluation protocol reports Parallel Turbo, and every number in the
model cards' main tables uses it. Operationally, though, the exa-auto
backend is the better default for NextSearch-1-S and -XS: it raised their
four-bench means by roughly +10pp in our sweeps (S 0.627→0.738,
XS 0.592→0.693, all benches up) at ~1.3× episode cost, because larger
result payloads replace extra search turns. NextSearch-1-M is nearly
backend-insensitive. Pick the backend with `--search-mode`; the
measurement methodology is in [evals.md](evals.md).
