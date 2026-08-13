"""NextSearch-1: the research-agent harness and evaluation suite.

Two entry points matter for most uses:

    from nextsearch.harness import run_episode      # one agent episode
    from nextsearch import harnesses                # prompt layer + toolset

and the CLI, `nextsearch-eval`, which runs the staged
prepare -> rollout -> grade -> report pipeline over a benchmark x model
matrix. See docs/evals.md.
"""

__version__ = "1.0.0"
