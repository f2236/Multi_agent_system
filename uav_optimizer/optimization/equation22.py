"""
Optimization problem formulation: Equation 22.

Maximization problem:
max R_S(s_T+2^S, k_T+2^S) = sum_u sum_m theta_{m,u,T+2} * r_{m,u,T+2}(s_{m,T+2}, k_{m,T+2})

subject to:
  - Equation 23: k_{n,t}, k_{m,t} in {1, ..., K} (antenna selection) — fixed to K=1
  - Equation 24: d_min <= d_{n,n',t}(s_{n,t}, s_{n',t}) (minimum UAV separation)
  - UAV dynamics: s_{n,t+1} = s_{n,t} + V*Δt*direction (Equation 20)
"""

from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
from uav_optimizer.domain import (
    UAVState,
    AntennaConfig,
    SimulationConfig,
    ObjectiveResult,
    Position3D,
)
from uav_optimizer.optimization.objective import ObjectiveEvaluator


@dataclass
class OptimizationProblem:
    """Formalization of the UAV trajectory optimization problem."""

    config: SimulationConfig
    antennas: List[AntennaConfig]
    num_uavs: int
    objective_evaluator: ObjectiveEvaluator

    # Problem parameters (from paper)
    K: int = 1  # Number of antenna ports (fixed to 1)
    T: int = 10  # Number of time steps before stable stage
    stable_stage: int = None  # T + 2

    def __post_init__(self):
        """Initialize problem parameters."""
        if self.stable_stage is None:
            self.stable_stage = self.T + 2

        # Validate
        if self.K != 1:
            raise ValueError("Only K=1 (single antenna port) is supported")

        if self.config.antenna_ports_per_uav != self.K:
            raise ValueError(
                f"Config antenna_ports_per_uav {self.config.antenna_ports_per_uav} "
                f"!= problem K {self.K}"
            )

    def objective(self, uav_states: List[UAVState]) -> float:
        """
        Compute objective function value (Equation 22).

        Args:
            uav_states: UAV states at stable stage

        Returns:
            R_S value (sum data rate in bps)
        """
        result = self.objective_evaluator.evaluate(uav_states, self.antennas)
        return result.value

    def constraint_violations(self, uav_states: List[UAVState]) -> Tuple[bool, List[str]]:
        """
        Check if state violates any constraints (Equations 23, 24).

        Args:
            uav_states: UAV states to validate

        Returns:
            (is_feasible, list_of_violations)
        """
        result = self.objective_evaluator.evaluate(uav_states, self.antennas)
        return result.constraints_satisfied, result.constraint_violations

    def get_problem_info(self) -> dict:
        """
        Get summary of the optimization problem.

        Returns:
            Dictionary with problem parameters
        """
        return {
            "problem_type": "UAV Trajectory Optimization (Stable Stage)",
            "objective": "Maximize R_S (sum data rate)",
            "equation": 22,
            "num_uavs": self.num_uavs,
            "num_antennas": len(self.antennas),
            "num_time_steps": self.T,
            "stable_stage_slot": self.stable_stage,
            "K": self.K,
            "antenna_port_selection": "Fixed (K=1)",
            "uav_speed": self.config.uav_speed,
            "time_step_duration": self.config.time_step_duration,
            "min_uav_separation": self.config.min_uav_separation,
            "max_uav_separation": self.config.max_uav_separation,
            "grid_bounds_x": self.config.grid_x_bounds,
            "grid_bounds_y": self.config.grid_y_bounds,
            "altitude_bounds": (self.config.min_altitude, self.config.max_altitude),
        }


class EquationValidator:
    """Validates problem equations and constraint satisfaction."""

    @staticmethod
    def validate_equation_20(
        old_position: Position3D,
        new_position: Position3D,
        direction_magnitude: float,
        max_speed: float,
        time_delta: float,
    ) -> Tuple[bool, str]:
        """
        Validate UAV movement satisfies Equation 20.

        s_{n,t+1} = s_{n,t} + V*Δt*direction

        Args:
            old_position: Position at time t
            new_position: Position at time t+1
            direction_magnitude: Magnitude of direction vector
            max_speed: UAV maximum speed (V)
            time_delta: Time step (Δt)

        Returns:
            (is_valid, message)
        """
        # Compute movement distance
        dx = new_position.x - old_position.x
        dy = new_position.y - old_position.y
        dz = new_position.z - old_position.z
        movement = (dx**2 + dy**2 + dz**2) ** 0.5

        # Expected maximum movement
        expected_max = max_speed * time_delta

        if movement > expected_max * 1.01:  # Allow 1% tolerance
            return False, (
                f"Movement {movement:.2f}m exceeds limit {expected_max:.2f}m "
                f"(V*Δt = {max_speed:.1f}*{time_delta:.1f})"
            )

        return True, ""

    @staticmethod
    def validate_equation_23(
        uav_states: List[UAVState],
        K: int = 1,
    ) -> Tuple[bool, List[str]]:
        """
        Validate antenna port selection satisfies Equation 23.

        k_{n,t}, k_{m,t} ∈ {1, ..., K}

        Args:
            uav_states: UAV states
            K: Number of antenna ports

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        for uav in uav_states:
            if not (1 <= uav.antenna_port <= K):
                violations.append(
                    f"UAV {uav.uav_id} antenna port {uav.antenna_port} "
                    f"not in {{1, ..., {K}}}"
                )

        return len(violations) == 0, violations

    @staticmethod
    def validate_equation_24(
        uav_states: List[UAVState],
        d_min: float,
        d_max: float,
    ) -> Tuple[bool, List[str]]:
        """
        Validate UAV separation satisfies Equation 24.

        d_min <= d_{n,n'}(s_n, s_n') <= d_max

        Args:
            uav_states: UAV states
            d_min: Minimum separation
            d_max: Maximum separation

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        for i in range(len(uav_states)):
            for j in range(i + 1, len(uav_states)):
                dist = uav_states[i].distance_to(uav_states[j])

                if dist < d_min:
                    violations.append(
                        f"UAV {uav_states[i].uav_id} and {uav_states[j].uav_id} "
                        f"separation {dist:.2f}m < d_min {d_min:.2f}m"
                    )

                if dist > d_max:
                    violations.append(
                        f"UAV {uav_states[i].uav_id} and {uav_states[j].uav_id} "
                        f"separation {dist:.2f}m > d_max {d_max:.2f}m"
                    )

        return len(violations) == 0, violations
