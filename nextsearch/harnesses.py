"""Harness registry: what the agent GETS — the prompt layer plus the toolset.

The episode loop in `harness.py` is prompt- and tool-agnostic, so a harness is
pure configuration: one entry, no behavior. Benchmarks declare that they are
agentic; the harness decides what an agentic benchmark actually receives.

Two harnesses ship, and they correspond to the two deployment shapes reported
for NextSearch-1:

  solo          a single research agent with `search` and `fetch`, plus the
                web-research guidelines. This is the configuration the models
                were trained in and the one every headline number uses.
  orchestrated  an orchestrator model that gets ONLY a `research` tool — each
                call runs a complete nested `solo` episode on a research
                subagent — plus orchestration guidelines. Used for the wide
                table tasks, where one research context is a poor fit for the
                breadth of the job.

Orthogonal to the harness is the task-date layer. Every system prompt is
prefixed with "Today's date is X." as its own paragraph, before the
guidelines: world state beside the task instruction, not research method.
Without it a model resolves "current", "latest", and "this year" from its
pretraining priors or from whatever dates happen to appear in results, and
freshness-sensitive accuracy drops substantially. The date is protocol — it
changes the prompt, so it changes the number — and is frozen separately in
each run manifest. `harness_prompt_sha` hashes the guidelines DOCUMENT only,
so it stays stable across dates.
"""

import hashlib
from dataclasses import dataclass

from .prompts import load_doc

# Under the orchestrated harness the orchestrator's output IS the deliverable,
# and wide tables run to hundreds of rows. The usual 16k completion cap would
# truncate the answer itself, so orchestrator sampling gets its own floor.
ORCHESTRATOR_MAX_TOKENS = 32768


@dataclass(frozen=True)
class SubagentDefaults:
    """Registry-level defaults for the research subagent. `doc` is the
    subagent's role prompt; `harness` supplies its guidelines and toolset."""
    model: str = "deepseek-v4-flash"
    harness: str = "solo"
    max_turns: int = 10
    doc: str = "subagent.md"


@dataclass(frozen=True)
class Harness:
    name: str
    guidelines_doc: "str | None"   # markdown file in prompts/
    tool_names: tuple              # toolset given to agentic benchmarks
    subagent: "SubagentDefaults | None" = None

    def system_suffix(self, today=None, tool_alias=None) -> "str | None":
        """The layer appended to every system prompt: an optional task-date
        line ("Today's date is YYYY-MM-DD.") followed by this harness's
        guidelines document.

        `tool_alias` rewrites the document's BACKTICKED tool symbols
        (`search` -> `web_search`) so the guidance stays consistent with
        renamed specs. Plain-prose uses of the same words ("searching", "a
        fetch") describe the capability rather than the symbol and are left
        alone.
        """
        doc = load_doc(self.guidelines_doc) if self.guidelines_doc else None
        if doc and tool_alias:
            for old, new in tool_alias.items():
                doc = doc.replace(f"`{old}`", f"`{new}`")
        parts = [f"Today's date is {today}." if today else None, doc]
        return "\n\n".join(p for p in parts if p) or None

    def tools(self, search_mode=None, subagent_run=None,
              fetch_window=None, tool_alias=None) -> list:
        """Build this harness's toolset.

        `subagent_run` (tools.subagent.SubagentRun) wires the `research`
        tool and is required when the toolset contains it — those tools are
        minted per (benchmark, model) pair rather than once per run, because
        each needs its own place to record nested rollouts. `fetch_window`
        overrides the fetch tool's window size. `tool_alias` ({old: new})
        renames tool SPECS after construction — same executor, new name, since
        dispatch keys off the spec name. An unknown old-name is an error, not
        a silent no-op.
        """
        from .tools import get_tools
        out = []
        for n in self.tool_names:
            if n == "research":
                if subagent_run is None:
                    raise ValueError(
                        f"harness {self.name!r} needs subagent_run wiring "
                        "for its research tool (see tools/subagent.py)")
                from .tools import subagent
                out.append(subagent.tool(subagent_run))
            elif n == "fetch" and fetch_window:
                from .tools import fetch
                out.append(fetch.tool(window_chars=fetch_window))
            else:
                out.extend(get_tools((n,), search_mode=search_mode))
        if tool_alias:
            names = {t.spec["name"] for t in out}
            unknown = set(tool_alias) - names
            if unknown:
                raise ValueError(
                    f"tool_alias renames unknown tools {sorted(unknown)}; "
                    f"harness {self.name!r} toolset: {sorted(names)}")
            from dataclasses import replace
            # Copy the spec dict — the tool modules share module-level SPEC
            # constants that must never be mutated in place.
            out = [replace(t, spec={**t.spec, "name": tool_alias[t.spec["name"]]})
                   if t.spec["name"] in tool_alias else t for t in out]
        return out

    def config(self) -> dict:
        """The manifest's protocol fragment: harness name plus a content hash
        of the guidelines, so editing the document visibly changes the
        evaluation. The hash covers the DOCUMENT, not the composed suffix, so
        it is stable across task dates (which are frozen separately). Toolset
        provenance lives in the manifest's top-level `tools` block."""
        cfg = {"harness": self.name}
        if self.guidelines_doc:
            cfg["harness_prompt_sha"] = hashlib.sha256(
                load_doc(self.guidelines_doc).encode()).hexdigest()[:12]
        return cfg


HARNESSES = {h.name: h for h in [
    Harness("solo", "web_research.md", ("search", "fetch")),
    Harness("orchestrated", "orchestrator.md", ("research",),
            subagent=SubagentDefaults()),
]}

DEFAULT_HARNESS = "solo"


def get(name) -> Harness:
    if name not in HARNESSES:
        raise KeyError(
            f"unknown harness {name!r}; available: {sorted(HARNESSES)}")
    return HARNESSES[name]
