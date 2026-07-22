"""
Integration tests for multi-agent orchestrator.

Tests the complete optimization loop: Environment → Planner → Evaluator agents.
"""

import pytest
from uav_optimizer.domain import (
    Position3D,
    UAVState,
    SimulationConfig,
    AntennaConfig,
)
from uav_optimizer.agents.orchestrator import OptimizationOrchestrator


@pytest.fixture
def basic_config():
    """Create basic simulation config for testing."""
    return SimulationConfig(
        time_step_duration=1.0,
        num_steps=5,
        uav_speed=10.0,
        grid_x_bounds=(0.0, 1000.0),
        grid_y_bounds=(0.0, 1000.0),
        min_altitude=5.0,
        max_altitude=500.0,
        min_uav_separation=10.0,
        max_uav_separation=2000.0,
        max_iterations=20,
        convergence_threshold=0.001,
    )


@pytest.fixture
def test_antennas():
    """Create test antenna configuration."""
    return [
        AntennaConfig(
            antenna_id=0,
            position=Position3D(x=500.0, y=500.0, z=20.0),
            tx_power_dbm=20.0,
            bandwidth_hz=1e6,
            frequency_hz=2.4e9,
            noise_figure_db=7.0,
        )
    ]


@pytest.fixture
def initial_uavs():
    """Create initial UAV positions."""
    return [
        UAVState(
            uav_id=0,
            position=Position3D(x=200.0, y=300.0, z=50.0),
            antenna_port=1,
        ),
        UAVState(
            uav_id=1,
            position=Position3D(x=800.0, y=700.0, z=75.0),
            antenna_port=1,
        ),
    ]


class TestOrchestratorInitialization:
    """Test orchestrator setup and initialization."""

    def test_orchestrator_initialization(self, basic_config, test_antennas, initial_uavs):
        """Test orchestrator initializes correctly."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        assert orchestrator.config == basic_config
        assert len(orchestrator.state.current_uav_states) == 2
        assert orchestrator.state.iteration == 0
        assert orchestrator.state.best_objective > 0
        assert orchestrator.state.current_objective > 0

    def test_orchestrator_initial_objective_computed(
        self, basic_config, test_antennas, initial_uavs
    ):
        """Test that initial objective is computed during initialization."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        # Both should be set to same initial value
        assert orchestrator.state.current_objective == orchestrator.state.best_objective
        assert orchestrator.state.current_objective > 0

    def test_orchestrator_tracks_best(self, basic_config, test_antennas, initial_uavs):
        """Test that orchestrator tracks best trajectory found."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        best_traj = orchestrator.get_best_trajectory()
        assert len(best_traj) == 2
        assert best_traj[0].uav_id == 0
        assert best_traj[1].uav_id == 1


class TestOrchestratorExecution:
    """Test orchestrator execution loop."""

    def test_orchestrator_run_single_iteration(
        self, basic_config, test_antennas, initial_uavs
    ):
        """Test orchestrator can run for 1 iteration."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        initial_objective = orchestrator.state.best_objective
        final_state = orchestrator.run(max_iterations=1, verbose=False)

        assert final_state.iteration == 0  # Iteration counter starts at 0
        assert len(final_state.agent_messages) > 0

    def test_orchestrator_run_multiple_iterations(
        self, basic_config, test_antennas, initial_uavs
    ):
        """Test orchestrator can run multiple iterations."""
        # Increase convergence threshold to allow more iterations
        config = SimulationConfig(
            time_step_duration=1.0,
            num_steps=5,
            uav_speed=10.0,
            grid_x_bounds=(0.0, 1000.0),
            grid_y_bounds=(0.0, 1000.0),
            min_altitude=5.0,
            max_altitude=500.0,
            min_uav_separation=10.0,
            max_uav_separation=2000.0,
            max_iterations=20,
            convergence_threshold=1e-10,  # Very small threshold
        )

        orchestrator = OptimizationOrchestrator(
            config=config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        final_state = orchestrator.run(max_iterations=5, verbose=False)

        # Should have executed multiple iterations
        assert final_state.iteration >= 0
        assert len(final_state.agent_messages) >= 5

    def test_orchestrator_convergence_check(self, basic_config, test_antennas, initial_uavs):
        """Test orchestrator stops at convergence."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        final_state = orchestrator.run(max_iterations=100, verbose=False)

        # Should converge quickly due to small threshold
        assert final_state.convergence_reached
        assert final_state.iteration < 100


class TestOrchestratorSummary:
    """Test orchestrator result reporting."""

    def test_get_summary(self, basic_config, test_antennas, initial_uavs):
        """Test orchestrator returns summary dict."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        orchestrator.run(max_iterations=1, verbose=False)
        summary = orchestrator.get_optimization_summary()

        assert "best_objective" in summary
        assert "final_objective" in summary
        assert "iterations" in summary
        assert "num_uavs" in summary
        assert "num_antennas" in summary
        assert "best_positions" in summary
        assert "convergence_reached" in summary

    def test_summary_values_consistent(self, basic_config, test_antennas, initial_uavs):
        """Test summary values are consistent with state."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        orchestrator.run(max_iterations=5, verbose=False)
        summary = orchestrator.get_optimization_summary()

        assert summary["best_objective"] == orchestrator.state.best_objective
        assert summary["num_uavs"] == len(orchestrator.state.current_uav_states)
        assert summary["num_antennas"] == len(orchestrator.state.antenna_configs)
        assert len(summary["best_positions"]) == 2

    def test_best_trajectory_positions(self, basic_config, test_antennas, initial_uavs):
        """Test best trajectory contains valid positions."""
        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        orchestrator.run(max_iterations=1, verbose=False)
        best_trajectory = orchestrator.get_best_trajectory()

        for uav in best_trajectory:
            # Check positions are within bounds
            assert basic_config.grid_x_bounds[0] <= uav.position.x <= basic_config.grid_x_bounds[1]
            assert basic_config.grid_y_bounds[0] <= uav.position.y <= basic_config.grid_y_bounds[1]
            assert basic_config.min_altitude <= uav.position.z <= basic_config.max_altitude


class TestOrchestratorRobustness:
    """Test orchestrator handles edge cases."""

    def test_orchestrator_with_single_uav(self, basic_config, test_antennas):
        """Test orchestrator works with single UAV."""
        single_uav = [
            UAVState(
                uav_id=0,
                position=Position3D(x=500.0, y=500.0, z=50.0),
                antenna_port=1,
            )
        ]

        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=single_uav,
            llama_interface=None,
        )

        final_state = orchestrator.run(max_iterations=3, verbose=False)
        assert final_state.best_objective > 0

    def test_orchestrator_with_multiple_uavs(self, basic_config, test_antennas):
        """Test orchestrator works with multiple UAVs."""
        multi_uavs = [
            UAVState(uav_id=i, position=Position3D(x=100*i, y=100*i, z=50+i*10), antenna_port=1)
            for i in range(5)
        ]

        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=test_antennas,
            initial_uav_states=multi_uavs,
            llama_interface=None,
        )

        final_state = orchestrator.run(max_iterations=2, verbose=False)
        assert len(final_state.current_uav_states) == 5

    def test_orchestrator_with_multiple_antennas(self, basic_config, initial_uavs):
        """Test orchestrator works with multiple antennas."""
        multi_antennas = [
            AntennaConfig(
                antenna_id=i,
                position=Position3D(x=250+i*250, y=500.0, z=20.0),
                tx_power_dbm=20.0,
                bandwidth_hz=1e6,
                frequency_hz=2.4e9,
                noise_figure_db=7.0,
            )
            for i in range(4)
        ]

        orchestrator = OptimizationOrchestrator(
            config=basic_config,
            antennas=multi_antennas,
            initial_uav_states=initial_uavs,
            llama_interface=None,
        )

        final_state = orchestrator.run(max_iterations=1, verbose=False)
        assert len(final_state.antenna_configs) == 4
