"""Agent evaluation: layered, rule-first, auditable.

``eval`` is deliberately a separate top-level package, not a corner of
``src/evaluation``: it *measures* the product and must not become importable
state the product depends on.  Only ``outcomes`` and ``report`` are pure
(stdlib, no project imports) so the metrics can be unit tested without Chroma,
torch or an answer model.

Layers:

    L1 retrieval   src/evaluation/retrieval_probe.py  (separate command, no LLM)
    L2 trajectory  eval.agent_eval  -- tool_trace rounds, gate, step budget
    L3 answer      eval.agent_eval  -- citation (file, page), numbers, refusals
    L4 refusal     eval.agent_eval  -- expect: answer | deny | refuse

Read ``eval/README.md`` before adding a metric: every ratio in a summary has to
carry its own denominator.
"""

from __future__ import annotations

# Single source of truth: the metrics module owns the version, so a summary can
# never claim a harness version the rules did not come from.
from eval.outcomes import HARNESS_VERSION

__all__ = ["HARNESS_VERSION"]