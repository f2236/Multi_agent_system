"""
Unit tests for optimization module.

Tests objective evaluation and equation validation.
"""

import pytest
from uav_optimizer.domain import (
    Position3D,
    UAVState,
    AntennaConfig,
    SimulationConfig,
)
from uav_optimizer.optimization.objective import ObjectiveEvaluator
from uav_optimizer.optimization.equation22 import OptimizationProblem, EquationValidator


class TestObjectiveEvaluator:
    """Test objective function evaluation (Equation 22)."""

    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        config = SimulationConfig(
            num_antennas=1,
            antenna_ports_per_uav=1,
        )
        antennas = [
            AntennaConfig(
                antenna_id=0,
                position=Position3D(x=500.0, y=500.0, z=20.0),
                tx_power_dbm=20.0,
                bandwidth_hz=1e6,
                frequency_hz=2.4e9,
                noise_figure_db=7.0,
            )
        ]
        evaluator = ObjectiveEvaluator(config)
        return config, antennas, evaluator

    def test_objective_single_uav_centered(self, setup):
        """Objective should be high when UAV is near antenna."""
        config, antennas, evaluator = setup

        # UAV very close to antenna
        uav = UAVState(
            uav_id=0,
            position=Position3D(x=500.0, y=500.0, z=30.0),  # Close to antenna
            antenna_port=1,
        )

        result = evaluator.evaluate([uav], antennas)
        assert result.value > 0
        assert result.constraints_satisfied

    def test_objective_single_uav_far(self, setup):
        """Objective should decrease when UAV is far from antenna."""
        config, antennas, evaluator = setup

        # UAV far from antenna
        uav = UAVState(
            uav_id=0,
            position=Position3D(x=0.0, y=0.0, z=100.0),  # Far from antenna
            antenna_port=1,
        )

        result = evaluator.evaluate([uav], antennas)
        # Should still be positive (Shannon capacity > 0 for any distance)
        assert result.value >= 0

    def test_objective_multiple_uavs(self, setup):
        """Objective should sum rates from multiple UAVs."""
        config, antennas, evaluator = setup

        uav1 = UAVState(
            uav_id=0,
            position=Position3D(x=500.0, y=500.0, z=30.0),
            antenna_port=1,
        )
        uav2 = UAVState(
            uav_id=1,
            position=Position3D(x=510.0, y=500.0, z=30.0),
            antenna_port=1,
        )

        result = evaluator.evaluate([uav1, uav2], antennas)
        assert result.value > 0
        assert result.constraints_satisfied
        # Should have 2 links evaluated
        assert len(result.per_link_rates) == 2

    def test_objective_constraint_violation(self, setup):
        """Objective should return 0 if constraints violated."""
        config, antennas, evaluator = setup

        # UAV below minimum altitude
        uav = UAVState(
            uav_id=0,
            position=Position3D(x=500.0, y=500.0, z=2.0),  # Below min_altitude=5.0
            antenna_port=1,
        )

        result = evaluator.evaluate([uav], antennas)
        assert result.value == 0.0
        assert not result.constraints_satisfied
        assert len(result.constraint_violations) > 0

    def test_objective_best_link_rate(self, setup):
        """Test getting best link rate."""
        config, antennas, evaluator = setup

        uav1 = UAVState(
            uav_id=0,
            position=Position3D(x=500.0, y=500.0, z=30.0),
            antenna_port=1,
        )
        uav2 = UAVState(
            uav_id=1,
            position=Position3D(x=0.0, y=0.0, z=100.0),
            antenna_port=1,
        )

        max_rate, best_link = evaluator.get_best_link_rate([uav1, uav2], antennas)
        assert max_rate > 0
        assert best_link[0] == 0  # antenna_id

    def test_objective_fairness_single_uav(self, setup):
        """Fairness metric for single UAV should be 1.0."""
        config, antennas, evaluator = setup

        uav = UAVState(
            uav_id=0,
            position=Position3D(x=500.0, y=500.0, z=30.0),
            antenna_port=1,
        )

        fairness = evaluator.get_fairness_metric([uav], antennas)
        assert abs(fairness - 1.0) < 0.01  # Single UAV = perfect fairness


class TestEquationValidator:
    """Test validation of equations and constraints."""

    def test_validate_equation_20_valid_movement(self):
        """Valid movement should pass Equation 20."""
        old_pos = Position3D(x=0.0, y=0.0, z=50.0)
        new_pos = Position3D(x=10.0, y=0.0, z=50.0)

        valid, msg = EquationValidator.validate_equation_20(
            old_pos, new_pos, direction_magnitude=1.0, max_speed=20.0, time_delta=1.0
        )
        assert valid

    def test_validate_equation_20_too_fast(self):
        """Movement exceeding max speed should fail Equation 20."""
        old_pos = Position3D(x=0.0, y=0.0, z=50.0)
        new_pos = Position3D(x=100.0, y=0.0, z=50.0)

        valid, msg = EquationValidator.validate_equation_20(
            old_pos, new_pos, direction_magnitude=1.0, max_speed=5.0, time_delta=1.0
        )
        assert not valid
        assert "exceeds limit" in msg

    def test_validate_equation_23_valid_ports(self):
        """Valid antenna ports should pass Equation 23."""
        uav1 = UAVState(
            uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0), antenna_port=1
        )
        uav2 = UAVState(
            uav_id=1, position=Position3D(x=100.0, y=0.0, z=50.0), antenna_port=1
        )

        valid, violations = EquationValidator.validate_equation_23([uav1, uav2], K=1)
        assert valid
        assert len(violations) == 0

    def test_validate_equation_23_invalid_ports(self):
        """Invalid antenna ports should fail Equation 23."""
        uav = UAVState(
            uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0), antenna_port=2
        )

        valid, violations = EquationValidator.validate_equation_23([uav], K=1)
        assert not valid
        assert len(violations) > 0

    def test_validate_equation_24_valid_separation(self):
        """Valid UAV separation should pass Equation 24."""
        uav1 = UAVState(
            uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0), antenna_port=1
        )
        uav2 = UAVState(
            uav_id=1, position=Position3D(x=100.0, y=0.0, z=50.0), antenna_port=1
        )

        valid, violations = EquationValidator.validate_equation_24(
            [uav1, uav2], d_min=10.0, d_max=200.0
        )
        assert valid
        assert len(violations) == 0

    def test_validate_equation_24_too_close(self):
        """UAVs too close should fail Equation 24."""
        uav1 = UAVState(
            uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0), antenna_port=1
        )
        uav2 = UAVState(
            uav_id=1, position=Position3D(x=5.0, y=0.0, z=50.0), antenna_port=1
        )

        valid, violations = EquationValidator.validate_equation_24(
            [uav1, uav2], d_min=10.0, d_max=200.0
        )
        assert not valid
        assert len(violations) > 0


class TestOptimizationProblem:
    """Test optimization problem formulation."""

    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        config = SimulationConfig(num_antennas=1, antenna_ports_per_uav=1)
        antennas = [
            AntennaConfig(
                antenna_id=0,
                position=Position3D(x=500.0, y=500.0, z=20.0),
            )
        ]
        evaluator = ObjectiveEvaluator(config)
        problem = OptimizationProblem(
            config=config,
            antennas=antennas,
            num_uavs=2,
            objective_evaluator=evaluator,
        )
        return problem, config, antennas

    def test_problem_initialization(self, setup):
        """Problem should initialize correctly."""
        problem, config, antennas = setup
        assert problem.K == 1
        assert problem.T == 10
        assert problem.stable_stage == 12

    def test_problem_objective_evaluation(self, setup):
        """Problem should evaluate objective correctly."""
        problem, config, antennas = setup

        uav1 = UAVState(
            uav_id=0, position=Position3D(x=500.0, y=500.0, z=30.0), antenna_port=1
        )
        uav2 = UAVState(
            uav_id=1, position=Position3D(x=510.0, y=500.0, z=30.0), antenna_port=1
        )

        objective_value = problem.objective([uav1, uav2])
        assert objective_value > 0

    def test_problem_constraint_check(self, setup):
        """Problem should check constraints correctly."""
        problem, config, antennas = setup

        # Valid state
        uav = UAVState(
            uav_id=0, position=Position3D(x=500.0, y=500.0, z=30.0), antenna_port=1
        )
        feasible, violations = problem.constraint_violations([uav])
        assert feasible

        # Invalid state (too low)
        uav_invalid = UAVState(
            uav_id=0, position=Position3D(x=500.0, y=500.0, z=2.0), antenna_port=1
        )
        feasible, violations = problem.constraint_violations([uav_invalid])
        assert not feasible

    def test_problem_info(self, setup):
        """Problem should provide correct info."""
        problem, config, antennas = setup
        info = problem.get_problem_info()

        assert info["num_uavs"] == 2
        assert info["num_antennas"] == 1
        assert info["K"] == 1
        assert info["objective"] == "Maximize R_S (sum data rate)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
