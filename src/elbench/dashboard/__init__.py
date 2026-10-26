"""Visitor-facing ELBench runner.

Scoring and model requests stay in the benchmark runner, the OpenAI-compatible
client, and the judge router. This package only selects items, estimates cost,
and hosts that path behind an OpenRouter key.
"""

from elbench.dashboard.catalog import Catalog, load_catalog
from elbench.dashboard.leaderboard import (
    LeaderboardStore,
    build_comparison_table,
    build_pareto_frontier,
    compute_normalized_scores,
)
from elbench.dashboard.plan import Plan, build_plan
from elbench.dashboard.protocol import VisitorProtocol, build_visitor_protocol
from elbench.dashboard.run import execute_run

__all__ = [
    "Catalog",
    "LeaderboardStore",
    "Plan",
    "VisitorProtocol",
    "build_comparison_table",
    "build_pareto_frontier",
    "build_plan",
    "build_visitor_protocol",
    "compute_normalized_scores",
    "execute_run",
    "load_catalog",
]
