"""Compatibility wrapper for the Groq-backed Llama planner."""

from typing import List, Tuple

from uav_framework.agents.planner import GroqPlanner, PlanningError
from uav_framework.simulation.types import SimulationConfig


class LLMPlanner:
    """Backward-compatible facade over :class:`GroqPlanner`.

    This class no longer falls back to random movement. If Groq is unavailable,
    the underlying planner raises ``PlanningError`` so the dashboard can show
    the failure.
    """

    def __init__(self, config=None, **kwargs):
        self.planner = GroqPlanner(config or SimulationConfig(), **kwargs)

    @property
    def request_count(self) -> int:
        return self.planner.request_count

    def plan(self, uavs, antennas) -> List[Tuple[int, int, int]]:
        result = self.planner.plan(uavs, antennas)
        return result.directions


__all__ = ["LLMPlanner", "PlanningError"]
