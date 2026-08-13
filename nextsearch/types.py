"""Canonical data shapes: chat messages, tool specs, Row, Rollout.

Messages are plain dicts in the OpenAI chat-completions shape, with one
extension — `reasoning_content` on assistant messages, so a model's thinking
is stored rather than discarded:

    {"role": "system"|"user", "content": str}
    {"role": "assistant", "content": str,
     "reasoning_content": str | None,          # omitted when absent
     "tool_calls": [{"id": str, "type": "function",
                     "function": {"name": str, "arguments": <JSON string>}}]}
    {"role": "tool", "tool_call_id": str, "content": str}

Tool specs are stored flat — {"name", "description", "parameters"} — and the
OpenAI {"type": "function", "function": ...} wrapper is applied only at the
API boundary (`tools_to_openai`).

A `Row` is one evaluation task: the prompt messages plus a gold answer. A
`Rollout` is one complete episode: the full message list plus scoring and
provenance, self-contained so it can be re-graded offline without re-running
the agent.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

ROLES = ("system", "user", "assistant", "tool")


def system(content):
    return {"role": "system", "content": content}


def user(content):
    return {"role": "user", "content": content}


def assistant(content="", reasoning_content=None, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        m["reasoning_content"] = reasoning_content
    if tool_calls:
        m["tool_calls"] = list(tool_calls)
    return m


def tool(content, tool_call_id):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def tool_call(name, arguments, id=None):
    """arguments may be a dict (serialized here) or an already-JSON string;
    the stored form is always the OpenAI JSON-string convention."""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def tool_spec(name, description, parameters):
    return {"name": name, "description": description, "parameters": parameters}


def tool_call_args(tc) -> dict:
    """Parsed arguments of a canonical tool call."""
    return json.loads(tc["function"]["arguments"])


# ---------------------------------------------------------------------------
# OpenAI wire conversion
#
# Canonical messages already ARE the OpenAI shape; only three deltas need
# handling. `reasoning_content` is dropped on the way out unless explicitly
# requested — sending history thinking back is a per-model rendering policy,
# and most servers ignore or reject it. Null content normalizes to "". Tool
# specs gain and lose the function wrapper.


def to_openai_messages(messages, include_reasoning=False):
    out = []
    for m in messages:
        m = dict(m)
        if not include_reasoning:
            m.pop("reasoning_content", None)
        out.append(m)
    return out


def from_openai_message(m):
    """One API response message (dict or SDK object) -> canonical message."""
    if not isinstance(m, dict):
        m = {k: getattr(m, k, None)
             for k in ("role", "content", "reasoning_content", "tool_calls")}
    calls = []
    for tc in m.get("tool_calls") or []:
        if not isinstance(tc, dict):
            tc = {"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name,
                               "arguments": tc.function.arguments}}
        calls.append(tc)
    return assistant(m.get("content") or "",
                     reasoning_content=m.get("reasoning_content"),
                     tool_calls=calls or None)


def tools_to_openai(specs):
    return [{"type": "function", "function": dict(s)} for s in specs]


# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One task: prompt messages + the gold the grader scores against.

    `gold` is polymorphic — its shape selects the grader. `{"answer", ...}`
    is judged as a single answer, `{"answer", "answer_type"}` runs the
    set-answer autorater, and `{"table", "evaluation"}` runs the table
    grader. `tools` is empty here: tool specs attach at rollout time from the
    selected harness, so one prepared dataset serves every harness.
    """
    messages: list
    target: Any = None
    gold: Any = None
    meta: dict = field(default_factory=dict)
    id: "str | None" = None
    tools: list = field(default_factory=list)
    v: int = 1

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        """Tolerant: known fields read, unknown ignored, missing defaulted."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Rollout:
    """One complete episode: full conversation + bookkeeping + scores.

    The gold is embedded, which is what makes a rollout file re-gradable on
    its own: change a judge or fix a parser and every historical episode can
    be rescored without touching the web again.
    """
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)   # specs available in the episode
    sample_id: "str | None" = None              # the Row.id that produced this
    gold: Any = None
    model: "str | None" = None
    sampling: dict = field(default_factory=dict)
    pred: Any = None
    correct: "bool | None" = None
    scores: dict = field(default_factory=dict)
    reward: "float | None" = None
    n_turns: "int | None" = None                # assistant turns = LLM calls
    n_tokens: "int | None" = None
    truncated: "bool | None" = None
    error: "str | None" = None
    timing: dict = field(default_factory=dict)  # wall clock, per-tool cost/latency
    meta: dict = field(default_factory=dict)
    v: int = 1

    @property
    def completion(self) -> str:
        """All assistant text, joined."""
        return "\n".join(m.get("content") or "" for m in self.messages
                         if m.get("role") == "assistant")

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
