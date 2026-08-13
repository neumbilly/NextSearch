# The harness

The harness is everything around the weights: the system prompt, the tool
schemas, how tool results are rendered, the interaction loop, the resource
limits, and what happens when one of them is hit.

It is worth treating as a first-class part of the system rather than
scaffolding. Developing this harness against frozen models — before any
training — moved a nine-cell benchmark mean from 0.648 to 0.747. That is
comparable to the effect of a round of post-training, and it is invisible in
any table that reports only model names.

Everything here is configuration, hashed into each run's manifest, so a score
belongs to a complete system rather than to a checkpoint.

## Two tools

**`search`** takes one research objective and returns up to ten ranked
results as titles, URLs, and bounded excerpts.

**`fetch`** reads specified URLs against an objective and returns the relevant
content as markdown excerpts. Adding it was one of the larger single wins we
measured: reading a selected page replaces several rounds of re-searching, so
it improved comprehensive set-answer accuracy *and* lowered cost at the same
time.

Fetched content is windowed at 4,000 characters per result, with a truncation
marker naming the offset to continue from. A continuation must repeat the same
URLs and objective with that offset, which re-slices the same cached
extraction rather than re-billing a fresh one — so paging is deterministic and
the page cannot change underneath the model mid-read. A 4k window beat both
smaller and larger settings in a 4k–16k sweep: larger windows improved recall
on some set-answer tasks but increased freshness errors and produced long
latency tails.

An offset that arrives without a matching cached extraction is *ignored, with
an explanation*, rather than honored. An offset is only meaningful against the
extract that produced its marker; honoring a foreign one pages silently wrong
content or lands past the end and wastes the turn.

## Matching the tool to the search backend

Search APIs are tuned for different query formulations, so evaluating one
through another's interface measures the mismatch rather than the backend.

The tool NAME and the returned result format are held constant across
backends; only the parameter schema and the parameter descriptions change, and
each follows that provider's own documented guidance. Parallel receives a
research objective plus three diverse keyword queries; Exa receives one
self-contained natural-language query; Tavily receives a concise single-topic
query. The model never sees which backend it is talking to — backend choice is
harness configuration, which is what makes a backend comparison controlled.

This matters more than it sounds. Switching a trained model's search backend
at inference time moved its four-benchmark mean by up to 11 points, and also
changed its *behavior*: richer results reduced repeated searching, but larger
payloads produced a context-overflow tail on some models. Backend selection
changes accuracy, cost, turn count, and failure mode together.

## Give the agent a clock

Every system prompt is prefixed with `Today's date is YYYY-MM-DD.` as its own
paragraph, before the research guidelines — world state beside the task, not
research method.

Without it, "current", "latest", and "this year" resolve from pretraining
priors or from whatever dates happen to appear in search results. Adding the
line improved mean freshness accuracy by roughly 13 points and comprehensive
set-answer accuracy by roughly 10, while a saturated historical control stayed
flat. The guidelines tell the agent to use recency only for facts that can
change, which limits recency bias on historical questions.

The date is protocol, not decoration: it changes the prompt, so it changes the
number, and it is frozen separately in every run manifest.

## Budgets the model can see

An episode is bounded four ways: turns, context tokens, tool calls per turn,
and wall-clock time. Each bounds something the others cannot.

Turns bound actions. Context bounds tokens. Wall clock bounds *time*, which
neither of the others does — and which matters most in the orchestrated
setting, where a turn blocks on the slowest subagent in its batch and one
pathological episode can stall its parent. Calls-per-turn bounds a single
turn's context growth, which the context check structurally cannot see coming:
it is evaluated against the previous call's usage, before this turn's results
exist. Without it, a model that emits a very large batch of tool calls in one
completion buries itself in results before the context check runs again.

**The limits are told to the model.** A transcript review found that reaching
a limit with no answer written was the single largest source of lost episodes
— often with the answer already sitting in the transcript. So the harness
appends a soft note as the budget gets close and a firm one on the last turn,
attached to the turn's last tool result rather than injected as a mid-thread
system message.

**And when a limit is hit anyway, the episode gets one more call.** No tools,
just: write your final answer from what you already have. This can only add an
answer, never remove one — if the salvage call fails, the episode keeps its
original outcome. The stop reason still records the limit that was hit, and
the episode is still marked truncated: it really did run out, and the report
must keep showing that. What changed is that grading now finds an answer.

Together these cut truncated-with-no-answer episodes by roughly a factor of
four in a 180-episode diagnostic.

## Failures are environment feedback, not crashes

A tool error is fed back to the model as the tool result and recorded. A
model-call error is retried with backoff, because a single transport flake
otherwise ends an episode whose research was already finished. A completion
that comes back with neither content nor tool calls is retried rather than
accepted as a terminal answer, which would silently score the episode zero.

Costs stay honest through all of it. A call that was skipped or that raised
genuinely spent nothing, so its cost is recorded as a known zero rather than
as missing — omitting it would poison the episode's rollup to unknown and drop
it from the run's cost table entirely. A call cut off mid-flight *is* unknown,
and is recorded as such, so the episode's cost becomes a reported lower bound
rather than a fiction.

## The orchestrated harness

For wide table tasks the top-level model gets only a `research` tool. Each
call runs a complete nested episode on a subagent, under the same loop and the
same `solo` toolset — which is exactly the configuration a research model is
evaluated under standalone, so any model can serve as the subagent with no
special-casing.

The orchestrator sees only the subagent's final report, never its search
transcript. A failed nested episode returns an actionable note *plus whatever
it had already written*: most failures still hold useful partial work, and an
orchestrator told only "it failed" re-delegates the same slice several times
over.

This changes the research contract. A subagent must return a useful brief
within a bounded turn and time allocation, and the orchestrator decides how
many calls to issue and how to combine them. Solo benchmark strength does not
determine the nested ranking — which is why both are reported.
