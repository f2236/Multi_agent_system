"""
Environment agent: observes and validates system state.

Responsibilities:
- Read current UAV positions and antenna configuration
- Validate constraints
- Report system health and feasibility
"""

from typing import List
from uav_optimizer.agents.base_agent import BaseAgent
from uav_optimizer.domain import OptimizationState, UAVState, Position3D
from uav_optimizer.simulator.constraints import ConstraintValidator


class EnvironmentAgent(BaseAgent):
    """Observes and validates the optimization environment."""

    def __init__(self, config):
        """Initialize environment agent."""
        super().__init__(name="EnvironmentAgent")
        self.config = config

    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute environment observation and validation.

        Args:
            state: Current optimization state

        Returns:
            Updated state with validation results
        """
        # Validate current UAV states
        valid, violations = ConstraintValidator.validate_uav_states(
            state.current_uav_states, self.config
        )

        state.constraints_valid = valid
        state.constraint_violations = violations

        # Generate observation summary
        num_uavs = len(state.current_uav_states)
        num_antennas = len(state.antenna_configs)

        obs_msg = (
            f"Environment check: {num_uavs} UAVs, {num_antennas} antennas. "
            f"Constraints valid: {valid}. Violations: {len(violations)}"
        )
        self.log_execution(state, obs_msg)

        if violations:
            for v in violations[:3]:  # Log first 3 violations
                state.agent_messages.append({"agent": "EnvironmentAgent", "violation": v})

        return state
