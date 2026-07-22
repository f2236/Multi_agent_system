"""
Unit tests for simulator module.

Tests channel model, constraints, and UAV movement.
"""

import pytest
import math
from uav_optimizer.domain import (
    Position3D,
    Direction3D,
    UAVState,
    AntennaConfig,
    SimulationConfig,
)
from uav_optimizer.simulator.channel_model import ChannelModel, LinkEvaluator
from uav_optimizer.simulator.constraints import ConstraintValidator
from uav_optimizer.simulator.simulator import UAVSimulator


class TestChannelModel:
    """Test wireless channel model calculations."""

    def test_free_space_path_loss_distance_scaling(self):
        """Path loss should increase with distance (20 log10 rule)."""
        d1 = 100.0  # meters
        d2 = 1000.0  # 10x distance

        pl1 = ChannelModel.free_space_path_loss(d1)
        pl2 = ChannelModel.free_space_path_loss(d2)

        # Path loss scales as (10x distance)^2 = 100x loss
        expected_ratio = 100.0
        actual_ratio = pl2 / pl1
        assert abs(actual_ratio - expected_ratio) < 1.0, (
            f"Path loss ratio {actual_ratio} != expected {expected_ratio}"
        )

    def test_free_space_path_loss_negative_distance(self):
        """Should raise error for negative distance."""
        with pytest.raises(ValueError):
            ChannelModel.free_space_path_loss(-10.0)

    def test_compute_snr_positive(self):
        """SNR should be positive for valid inputs."""
        snr = ChannelModel.compute_snr(
            tx_power_w=1.0,
            distance=100.0,
            bandwidth_hz=1e6,
            noise_figure_db=7.0,
        )
        assert snr > 0

    def test_compute_data_rate_shannon(self):
        """Data rate should follow Shannon formula."""
        rate = ChannelModel.compute_data_rate(
            tx_power_w=1.0,
            distance=100.0,
            bandwidth_hz=1e6,
            noise_figure_db=7.0,
        )
        # Should be positive and less than Shannon limit
        assert 0 < rate < 1e6 * math.log2(1e10)


class TestConstraintValidator:
    """Test constraint validation."""

    def test_altitude_bounds_valid(self):
        """Valid altitude should pass."""
        config = SimulationConfig()
        pos = Position3D(x=500.0, y=500.0, z=100.0)
        valid, msg = ConstraintValidator.check_altitude_bounds(pos, config)
        assert valid

    def test_altitude_bounds_too_low(self):
        """Altitude below minimum should fail."""
        config = SimulationConfig()
        pos = Position3D(x=500.0, y=500.0, z=2.0)  # Below min_altitude=5.0
        valid, msg = ConstraintValidator.check_altitude_bounds(pos, config)
        assert not valid
        assert "below minimum" in msg

    def test_grid_bounds_valid(self):
        """Position within grid should pass."""
        config = SimulationConfig()
        pos = Position3D(x=500.0, y=500.0, z=100.0)
        valid, msg = ConstraintValidator.check_grid_bounds(pos, config)
        assert valid

    def test_grid_bounds_outside(self):
        """Position outside grid should fail."""
        config = SimulationConfig()
        pos = Position3D(x=2000.0, y=500.0, z=100.0)  # Outside [0, 1000]
        valid, msg = ConstraintValidator.check_grid_bounds(pos, config)
        assert not valid

    def test_minimum_separation_valid(self):
        """UAVs with adequate separation should pass."""
        config = SimulationConfig(min_uav_separation=10.0)
        uav1 = UAVState(uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0))
        uav2 = UAVState(uav_id=1, position=Position3D(x=50.0, y=0.0, z=50.0))
        valid, violations = ConstraintValidator.check_minimum_separation(
            [uav1, uav2], config
        )
        assert valid

    def test_minimum_separation_violated(self):
        """UAVs too close should fail."""
        config = SimulationConfig(min_uav_separation=100.0)
        uav1 = UAVState(uav_id=0, position=Position3D(x=0.0, y=0.0, z=50.0))
        uav2 = UAVState(uav_id=1, position=Position3D(x=10.0, y=0.0, z=50.0))
        valid, violations = ConstraintValidator.check_minimum_separation(
            [uav1, uav2], config
        )
        assert not valid
        assert len(violations) > 0


class TestUAVSimulator:
    """Test UAV movement simulation."""

    def test_simulator_initialization(self):
        """Simulator should initialize correctly."""
        config = SimulationConfig()
        sim = UAVSimulator(config)
        assert sim.current_time_slot == 0

    def test_movement_with_zero_direction(self):
        """Zero direction should result in no movement."""
        config = SimulationConfig()
        sim = UAVSimulator(config)

        uav = UAVState(uav_id=0, position=Position3D(x=100.0, y=100.0, z=50.0))
        direction = Direction3D(dx=0, dy=0, dz=0)

        new_states, violations = sim.step([uav], [direction])

        assert len(new_states) == 1
        assert new_states[0].position.x == uav.position.x
        assert new_states[0].position.y == uav.position.y

    def test_movement_with_positive_direction(self):
        """Positive direction should move UAV forward."""
        config = SimulationConfig(uav_speed=10.0, time_step_duration=1.0)
        sim = UAVSimulator(config)

        uav = UAVState(uav_id=0, position=Position3D(x=100.0, y=100.0, z=50.0))
        direction = Direction3D(dx=1, dy=0, dz=0)  # Move in +X direction

        new_states, violations = sim.step([uav], [direction])

        assert len(new_states) == 1
        assert new_states[0].position.x > uav.position.x
        assert new_states[0].position.y == uav.position.y

    def test_altitude_constraint_enforcement(self):
        """Simulator should enforce altitude constraints."""
        config = SimulationConfig(min_altitude=5.0, max_altitude=500.0)
        sim = UAVSimulator(config)

        # Try to move below minimum altitude
        uav = UAVState(uav_id=0, position=Position3D(x=100.0, y=100.0, z=10.0))
        direction = Direction3D(dx=0, dy=0, dz=-1)

        new_states, violations = sim.step([uav], [direction])

        # Altitude should be clamped to minimum
        assert new_states[0].position.z >= config.min_altitude

    def test_time_slot_increment(self):
        """Time slot should increment after each step."""
        config = SimulationConfig()
        sim = UAVSimulator(config)

        uav = UAVState(uav_id=0, position=Position3D(x=100.0, y=100.0, z=50.0))
        direction = Direction3D(dx=0, dy=0, dz=0)

        assert sim.current_time_slot == 0
        sim.step([uav], [direction])
        assert sim.current_time_slot == 1
        sim.step([uav], [direction])
        assert sim.current_time_slot == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
