"""
LangGraph orchestrator for multi-agent optimization loop.

Coordinates Environment → Planner → Evaluator agents in iterative optimization.
"""

from typing import List, Optional, Callable, Any
from uav_optimizer.domain import OptimizationState, SimulationConfig, UAVState, AntennaConfig, Position3D
from uav_optimizer.agents.base_agent import BaseAgent
from uav_optimizer.agents.environment import EnvironmentAgent
from uav_optimizer.agents.planner import PlannerAgent
from uav_optimizer.agents.evaluator import EvaluatorAgent
from uav_optimizer.optimization.objective import ObjectiveEvaluator


class OptimizationOrchestrator:
    """Orchestrates multi-agent optimization loop using LangGraph pattern."""

    def __init__(
        self,
        config: SimulationConfig,
        antennas: List[AntennaConfig],
        initial_uav_states: List[UAVState],
        llama_interface: Optional[Any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Simulation configuration
            antennas: List of antenna configurations
            initial_uav_states: Initial UAV positions
            llama_interface: Optional Llama interface
        """
        self.config = config
        self.antennas = antennas
        self.initial_uav_states = initial_uav_states

        # Initialize objective evaluator
        self.objective_evaluator = ObjectiveEvaluator(config)

        # Initialize agents
        self.environment_agent = EnvironmentAgent(config)
        self.planner_agent = PlannerAgent(config, llama_interface)
        self.evaluator_agent = EvaluatorAgent(config, self.objective_evaluator)

        # Initialize state
        self.state = OptimizationState(
            iteration=0,
            current_uav_states=list(initial_uav_states),
            antenna_configs=list(antennas),
            current_objective=0.0,
            best_objective=0.0,
            best_uav_states=list(initial_uav_states),
        )

        # Compute initial objective
        result = self.objective_evaluator.evaluate(self.state.current_uav_states, antennas)
        self.state.current_objective = result.value
        self.state.best_objective = result.value
        # Initialize objective history for convergence plotting
        self.state.objective_history = [result.value]

    def run(
        self,
        max_iterations: Optional[int] = None,
        convergence_check: Optional[Callable[[OptimizationState], bool]] = None,
        verbose: bool = True,
    ) -> OptimizationState:
        """
        Run multi-agent optimization loop.

        Agent sequence:
        1. EnvironmentAgent: Observe and validate state
        2. PlannerAgent: Propose new positions
        3. EvaluatorAgent: Evaluate and accept/reject

        Args:
            max_iterations: Maximum iterations (or use config value)
            convergence_check: Optional custom convergence check
            verbose: Print progress

        Returns:
            Final optimization state
        """
        max_iters = max_iterations or self.config.max_iterations

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting UAV Trajectory Optimization")
            print(f"{'='*70}")
            print(f"UAVs: {len(self.state.current_uav_states)}")
            print(f"Antennas: {len(self.state.antenna_configs)}")
            print(f"Max iterations: {max_iters}")
            print(f"Initial objective: {self.state.current_objective:.2e}")
            print(f"{'='*70}\n")

        for iteration in range(max_iters):
            self.state.iteration = iteration

            # 1. Environment agent: observe and validate
            self.state = self.environment_agent.execute(self.state)

            # 2. Planner agent: propose trajectories
            self.state = self.planner_agent.execute(self.state)

            # 3. Evaluator agent: evaluate and accept/reject
            self.state = self.evaluator_agent.execute(self.state)

            # Record objective history after evaluation
            try:
                self.state.objective_history.append(self.state.current_objective)
            except Exception:
                # Ensure attribute exists
                self.state.objective_history = [self.state.current_objective]

            # Print progress
            if verbose and iteration % max(1, max_iters // 10) == 0:
                print(
                    f"Iter {iteration:3d}: "
                    f"Current R_S = {self.state.current_objective:.2e}, "
                    f"Best R_S = {self.state.best_objective:.2e}"
                )

            # Check custom convergence
            if convergence_check and convergence_check(self.state):
                if verbose:
                    print(f"Convergence reached at iteration {iteration}")
                break

            # Check built-in convergence
            if self.state.convergence_reached:
                if verbose:
                    print(f"Convergence threshold reached at iteration {iteration}")
                break

        if verbose:
            print(f"\n{'='*70}")
            print(f"Optimization Complete")
            print(f"Final best objective: {self.state.best_objective:.2e}")
            print(f"Total iterations: {self.state.iteration + 1}")
            print(f"Total agent messages: {len(self.state.agent_messages)}")
            print(f"{'='*70}\n")

        return self.state

    def get_best_trajectory(self) -> List[UAVState]:
        """
        Get best trajectory found.

        Returns:
            List of UAV states at best objective
        """
        return self.state.best_uav_states

    def get_optimization_summary(self) -> dict:
        """
        Get summary of optimization results.

        Returns:
            Dictionary with results summary
        """
        return {
            "best_objective": self.state.best_objective,
            "final_objective": self.state.current_objective,
            "iterations": self.state.iteration + 1,
            "num_uavs": len(self.state.current_uav_states),
            "num_antennas": len(self.state.antenna_configs),
            "best_positions": [
                (u.uav_id, u.position.to_tuple()) for u in self.state.best_uav_states
            ],
            "convergence_reached": self.state.convergence_reached,
        }
