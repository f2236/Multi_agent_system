"""
Constraint validation for UAV trajectories.

Implements Equation 24 and other safety constraints.
"""

from typing import List, Tuple
from uav_optimizer.domain import UAVState, SimulationConfig


class ConstraintValidator:
    """Validates UAV positions and movements against constraints."""

    @staticmethod
    def check_altitude_bounds(
        position: "Position3D",  # type: ignore
        config: SimulationConfig,
    ) -> Tuple[bool, str]:
        """
        Check if altitude is within bounds.

        Args:
            position: UAV position
            config: Simulation configuration

        Returns:
            (is_valid, error_message)
        """
        if position.z < config.min_altitude:
            return False, f"Altitude {position.z} below minimum {config.min_altitude}"
        if position.z > config.max_altitude:
            return False, f"Altitude {position.z} above maximum {config.max_altitude}"
        return True, ""

    @staticmethod
    def check_grid_bounds(
        position: "Position3D",  # type: ignore
        config: SimulationConfig,
    ) -> Tuple[bool, str]:
        """
        Check if position is within grid bounds.

        Args:
            position: UAV position
            config: Simulation configuration

        Returns:
            (is_valid, error_message)
        """
        x_min, x_max = config.grid_x_bounds
        y_min, y_max = config.grid_y_bounds

        if not (x_min <= position.x <= x_max):
            return False, f"X coordinate {position.x} outside bounds [{x_min}, {x_max}]"
        if not (y_min <= position.y <= y_max):
            return False, f"Y coordinate {position.y} outside bounds [{y_min}, {y_max}]"

        return True, ""

    @staticmethod
    def check_minimum_separation(
        uav_states: List[UAVState],
        config: SimulationConfig,
    ) -> Tuple[bool, List[str]]:
        """
        Check if all UAVs maintain minimum separation.

        Equation 24: d_min <= d_{n,n'}(s_n, s_n') for all n != n'

        Args:
            uav_states: List of UAV states
            config: Simulation configuration

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        for i in range(len(uav_states)):
            for j in range(i + 1, len(uav_states)):
                dist = uav_states[i].distance_to(uav_states[j])

                if dist < config.min_uav_separation:
                    violations.append(
                        f"UAV {uav_states[i].uav_id} and {uav_states[j].uav_id} "
                        f"separation {dist:.2f}m < minimum {config.min_uav_separation}m"
                    )

                if dist > config.max_uav_separation:
                    violations.append(
                        f"UAV {uav_states[i].uav_id} and {uav_states[j].uav_id} "
                        f"separation {dist:.2f}m > maximum {config.max_uav_separation}m"
                    )

        return len(violations) == 0, violations

    @staticmethod
    def check_maximum_speed(
        old_position: "Position3D",  # type: ignore
        new_position: "Position3D",  # type: ignore
        time_delta: float,
        max_speed: float,
    ) -> Tuple[bool, str]:
        """
        Check if movement respects maximum speed constraint.

        Equation 20: new_pos = old_pos + V * Δt * direction

        Args:
            old_position: Previous position
            new_position: New position
            time_delta: Time step in seconds
            max_speed: Maximum speed in m/s

        Returns:
            (is_valid, error_message)
        """
        dx = new_position.x - old_position.x
        dy = new_position.y - old_position.y
        dz = new_position.z - old_position.z
        distance = (dx**2 + dy**2 + dz**2) ** 0.5

        implied_speed = distance / max(time_delta, 1e-6)

        if implied_speed > max_speed * 1.01:  # Allow 1% tolerance
            return False, (
                f"Implied speed {implied_speed:.2f} m/s exceeds "
                f"maximum {max_speed} m/s"
            )

        return True, ""

    @staticmethod
    def validate_position(
        position: "Position3D",  # type: ignore
        config: SimulationConfig,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single position against all constraints.

        Args:
            position: UAV position
            config: Simulation configuration

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        valid, msg = ConstraintValidator.check_altitude_bounds(position, config)
        if not valid:
            violations.append(msg)

        valid, msg = ConstraintValidator.check_grid_bounds(position, config)
        if not valid:
            violations.append(msg)

        return len(violations) == 0, violations

    @staticmethod
    def validate_uav_states(
        uav_states: List[UAVState],
        config: SimulationConfig,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all UAV states comprehensively.

        Args:
            uav_states: List of UAV states
            config: Simulation configuration

        Returns:
            (is_valid, list_of_violations)
        """
        all_violations = []

        # Check individual positions
        for uav in uav_states:
            valid, violations = ConstraintValidator.validate_position(
                uav.position, config
            )
            if not valid:
                all_violations.extend(
                    [f"UAV {uav.uav_id}: {v}" for v in violations]
                )

        # Check separation constraints
        valid, violations = ConstraintValidator.check_minimum_separation(
            uav_states, config
        )
        if not valid:
            all_violations.extend(violations)

        return len(all_violations) == 0, all_violations
