"""
Objective function evaluation: Equation 22.

Maximizes sum data rate at stable stage:
R_S(s_T+2^S, k_T+2^S) = sum_u sum_m theta_{m,u,T+2} * r_{m,u,T+2}(s_m,T+2, k_m,T+2)

where:
  - s_T+2^S: 3D location vector of UAVs at stable stage
  - k_T+2^S: antenna port selection vector (fixed, K=1)
  - theta: weighting factors (set to 1 for K=1)
  - r: individual link data rates
"""

from typing import List, Dict, Tuple
from uav_optimizer.domain import (
    Position3D,
    UAVState,
    AntennaConfig,
    ObjectiveResult,
    SimulationConfig,
)
from uav_optimizer.simulator.channel_model import LinkEvaluator
from uav_optimizer.simulator.constraints import ConstraintValidator
import time


class ObjectiveEvaluator:
    """Evaluates the sum data rate objective (Equation 22)."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize objective evaluator.

        Args:
            config: Simulation configuration
        """
        self.config = config

    def evaluate(
        self,
        uav_states: List[UAVState],
        antennas: List[AntennaConfig],
        iteration: int = 0,
    ) -> ObjectiveResult:
        """
        Evaluate sum data rate objective at stable stage.

        Equation 22:
        R_S = sum_{u=1}^U sum_{m in M} theta_{m,u,T+2} * r_{m,u,T+2}(s_{m,T+2}, k_{m,T+2})

        Args:
            uav_states: List of UAV states (at stable stage)
            antennas: List of antenna configurations
            iteration: Current iteration number

        Returns:
            ObjectiveResult with R_S value and metrics
        """
        result = ObjectiveResult(
            value=0.0,
            iteration=iteration,
            timestamp=time.time(),
            uav_positions=[u.position for u in uav_states],
            per_link_rates={},
            constraints_satisfied=True,
            constraint_violations=[],
        )

        # Validate constraints
        valid, violations = ConstraintValidator.validate_uav_states(uav_states, self.config)
        result.constraints_satisfied = valid
        result.constraint_violations = violations

        if not valid:
            # Return 0 objective if constraints violated
            result.value = 0.0
            return result

        # Compute per-link rates
        total_rate = 0.0
        link_count = 0

        for uav in uav_states:
            for antenna in antennas:
                # Evaluate link from UAV to antenna
                channel = LinkEvaluator.evaluate_link(uav.position, antenna)

                # Update channel with UAV ID
                channel.uav_id = uav.uav_id

                # Get theta weight (K=1, so theta=1 for all links)
                # In general: theta_{m,u,T+2}, but we set to 1
                theta = 1.0

                # Individual link rate: r_{m,u,T+2}
                link_rate = channel.data_rate_bps * theta

                # Store per-link rate
                link_key = (antenna.antenna_id, uav.uav_id)
                result.per_link_rates[link_key] = channel.data_rate_bps

                # Accumulate to sum
                total_rate += link_rate
                link_count += 1

        result.value = total_rate
        return result

    def evaluate_batch(
        self,
        trajectories: List[List[UAVState]],
        antennas: List[AntennaConfig],
    ) -> List[ObjectiveResult]:
        """
        Evaluate multiple trajectories (snapshots).

        Args:
            trajectories: List of UAV state snapshots
            antennas: List of antenna configurations

        Returns:
            List of ObjectiveResult, one per trajectory
        """
        results = []
        for idx, uav_states in enumerate(trajectories):
            result = self.evaluate(uav_states, antennas, iteration=idx)
            results.append(result)
        return results

    def get_best_link_rate(
        self,
        uav_states: List[UAVState],
        antennas: List[AntennaConfig],
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Get the best link rate among all UAV-antenna pairs.

        Args:
            uav_states: List of UAV states
            antennas: List of antenna configurations

        Returns:
            (max_rate_bps, (antenna_id, uav_id))
        """
        max_rate = 0.0
        best_link = (0, 0)

        for uav in uav_states:
            for antenna in antennas:
                channel = LinkEvaluator.evaluate_link(uav.position, antenna)
                if channel.data_rate_bps > max_rate:
                    max_rate = channel.data_rate_bps
                    best_link = (antenna.antenna_id, uav.uav_id)

        return max_rate, best_link

    def get_worst_link_rate(
        self,
        uav_states: List[UAVState],
        antennas: List[AntennaConfig],
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Get the worst link rate among all UAV-antenna pairs.

        Args:
            uav_states: List of UAV states
            antennas: List of antenna configurations

        Returns:
            (min_rate_bps, (antenna_id, uav_id))
        """
        min_rate = float("inf")
        worst_link = (0, 0)

        for uav in uav_states:
            for antenna in antennas:
                channel = LinkEvaluator.evaluate_link(uav.position, antenna)
                if channel.data_rate_bps < min_rate:
                    min_rate = channel.data_rate_bps
                    worst_link = (antenna.antenna_id, uav.uav_id)

        if min_rate == float("inf"):
            min_rate = 0.0

        return min_rate, worst_link

    def get_fairness_metric(
        self,
        uav_states: List[UAVState],
        antennas: List[AntennaConfig],
    ) -> float:
        """
        Compute fairness metric (Jain's fairness index).

        Metric: (sum R)^2 / (n * sum R^2)
        - 1.0 = perfectly fair
        - 0.0 = completely unfair

        Args:
            uav_states: List of UAV states
            antennas: List of antenna configurations

        Returns:
            Fairness index in [0, 1]
        """
        rates = []

        for uav in uav_states:
            uav_total_rate = 0.0
            for antenna in antennas:
                channel = LinkEvaluator.evaluate_link(uav.position, antenna)
                uav_total_rate += channel.data_rate_bps
            rates.append(uav_total_rate)

        if len(rates) == 0 or all(r == 0 for r in rates):
            return 0.0

        n = len(rates)
        sum_rates = sum(rates)
        sum_rates_sq = sum(r**2 for r in rates)

        if sum_rates_sq == 0:
            return 0.0

        fairness = (sum_rates**2) / (n * sum_rates_sq)
        return min(1.0, max(0.0, fairness))
