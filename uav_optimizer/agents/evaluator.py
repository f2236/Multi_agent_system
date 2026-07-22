"""
Evaluator agent: computes objectives and determines trajectory acceptance.

Responsibilities:
- Evaluate proposed positions using objective function
- Compare with current best
- Accept/reject based on acceptance policy
- Track convergence
"""

from uav_optimizer.agents.base_agent import BaseAgent
from uav_optimizer.domain import OptimizationState, UAVState
from uav_optimizer.optimization.objective import ObjectiveEvaluator


class EvaluatorAgent(BaseAgent):
    """Evaluates trajectories and makes acceptance decisions."""

    def __init__(
        self,
        config,
        objective_evaluator: ObjectiveEvaluator,
        acceptance_policy: str = "greedy",
    ):
        """
        Initialize evaluator agent.

        Args:
            config: Simulation configuration
            objective_evaluator: Objective function evaluator
            acceptance_policy: "greedy" or "simulated_annealing"
        """
        super().__init__(name="EvaluatorAgent")
        self.config = config
        self.objective_evaluator = objective_evaluator
        self.acceptance_policy = acceptance_policy
        self.temperature = 1.0  # For simulated annealing

    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute evaluation and acceptance decision.

        Args:
            state: Current optimization state

        Returns:
            Updated state with evaluation results
        """
        # Build proposed UAV states
        if not state.proposed_positions:
            self.log_execution(state, "No proposed positions to evaluate")
            return state

        proposed_states = [
            UAVState(
                uav_id=uav.uav_id,
                position=pos,
                antenna_port=uav.antenna_port,
            )
            for uav, pos in zip(state.current_uav_states, state.proposed_positions)
        ]

        # Evaluate proposed trajectory
        result = self.objective_evaluator.evaluate(proposed_states, state.antenna_configs)

        # Make acceptance decision
        accept = self._should_accept(
            current_value=state.current_objective,
            proposed_value=result.value,
            iteration=state.iteration,
        )

        if accept and result.constraints_satisfied:
            # Accept the proposal
            state.current_uav_states = proposed_states
            state.current_objective = result.value

            if result.value > state.best_objective:
                state.best_objective = result.value
                state.best_uav_states = proposed_states

            self.log_execution(
                state,
                f"ACCEPTED: R_S = {result.value:.2e} (new best: {state.best_objective:.2e})",
            )

        else:
            reason = (
                "constraint violation"
                if not result.constraints_satisfied
                else "worse objective"
            )
            self.log_execution(
                state,
                f"REJECTED ({reason}): R_S = {result.value:.2e} < current {state.current_objective:.2e}",
            )

        # Check convergence
        if state.iteration > 0:
            improvement = state.best_objective - state.current_objective
            if abs(improvement) < self.config.convergence_threshold:
                state.convergence_reached = True
                self.log_execution(state, "CONVERGENCE THRESHOLD REACHED")

        # Update temperature for SA
        self.temperature *= 0.95

        return state

    def _should_accept(
        self,
        current_value: float,
        proposed_value: float,
        iteration: int,
    ) -> bool:
        """
        Determine if proposed trajectory should be accepted.

        Args:
            current_value: Current objective value
            proposed_value: Proposed objective value
            iteration: Current iteration

        Returns:
            True if proposal should be accepted
        """
        if self.acceptance_policy == "greedy":
            return proposed_value >= current_value

        elif self.acceptance_policy == "simulated_annealing":
            # Accept if better, or with probability exp(-delta/T)
            if proposed_value >= current_value:
                return True

            import math
            import random

            delta = proposed_value - current_value  # negative
            acceptance_prob = math.exp(delta / max(self.temperature, 0.001))
            return random.random() < acceptance_prob

        else:
            # Default to greedy
            return proposed_value >= current_value
