"""
UAV environment simulator.

Implements Equation 20 (UAV movement model) and state evolution.
"""

import math
import random
from typing import List, Tuple, Optional
from copy import deepcopy
from uav_optimizer.domain import (
    Position3D,
    Direction3D,
    UAVState,
    SimulationConfig,
)
from uav_optimizer.simulator.constraints import ConstraintValidator


class UAVSimulator:
    """Simulates UAV dynamics and movement."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize UAV simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config
        random.seed(config.seed)
        self._time_slot = 0

    def step(
        self,
        uav_states: List[UAVState],
        directions: Optional[List[Direction3D]] = None,
    ) -> Tuple[List[UAVState], List[str]]:
        """
        Evolve UAV states by one time step.

        Equation 20: s_{n,t+1}(Δx, Δy, Δz) = s_{n,t} + V·Δt · [Δx, Δy, Δz]

        where:
          - Δx, Δy, Δz ∈ {-1, 0, +1}
          - V = flying speed (m/s)
          - Δt = time_step_duration (s)

        Args:
            uav_states: Current UAV states
            directions: Movement directions per UAV (if None, use heuristic)

        Returns:
            (updated_uav_states, constraint_violations)
        """
        if directions is None:
            # Heuristic: move toward grid center or random walk
            directions = self._generate_heuristic_directions(uav_states)

        # Validate input
        if len(directions) != len(uav_states):
            raise ValueError(
                f"directions length {len(directions)} != uav_states length {len(uav_states)}"
            )

        # Apply movement model to each UAV
        new_states = []
        for uav, direction in zip(uav_states, directions):
            new_pos = self._apply_movement(uav.position, direction)
            new_state = UAVState(
                uav_id=uav.uav_id,
                position=new_pos,
                antenna_port=uav.antenna_port,
            )
            new_states.append(new_state)

        # Validate new states
        valid, violations = ConstraintValidator.validate_uav_states(new_states, self.config)

        self._time_slot += 1

        return new_states, violations

    def _apply_movement(
        self,
        current_pos: Position3D,
        direction: Direction3D,
    ) -> Position3D:
        """
        Apply movement model (Equation 20).

        new_pos = old_pos + V·Δt · direction_unit_vector

        Args:
            current_pos: Current position
            direction: Movement direction in {-1, 0, +1}

        Returns:
            New position
        """
        # Maximum step distance per time slot
        step_magnitude = self.config.uav_speed * self.config.time_step_duration

        # Direction as unit vector
        direction_vec = [float(direction.dx), float(direction.dy), float(direction.dz)]
        norm = math.sqrt(sum(x**2 for x in direction_vec))

        if norm < 1e-6:
            # No movement (direction = (0, 0, 0))
            return current_pos

        # Normalize and scale
        direction_unit = [x / norm for x in direction_vec]
        step_vec = [step_magnitude * x for x in direction_unit]

        # Compute new position
        new_x = current_pos.x + step_vec[0]
        new_y = current_pos.y + step_vec[1]
        new_z = current_pos.z + step_vec[2]

        # Enforce altitude bounds
        new_z = max(self.config.min_altitude, min(new_z, self.config.max_altitude))

        # Enforce grid bounds (with wrapping or clamping)
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds

        # Clamp to grid (no wrapping)
        new_x = max(x_min, min(new_x, x_max))
        new_y = max(y_min, min(new_y, y_max))

        return Position3D(x=new_x, y=new_y, z=new_z)

    def _generate_heuristic_directions(
        self,
        uav_states: List[UAVState],
    ) -> List[Direction3D]:
        """
        Generate heuristic movement directions.

        Simple strategy: move toward grid center with occasional random jitter.

        Args:
            uav_states: Current UAV states

        Returns:
            List of directions
        """
        directions = []
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds
        grid_center_x = (x_min + x_max) / 2
        grid_center_y = (y_min + y_max) / 2

        for uav in uav_states:
            # Compute direction toward center
            dx_sign = 0
            if uav.position.x < grid_center_x - 50:
                dx_sign = 1
            elif uav.position.x > grid_center_x + 50:
                dx_sign = -1

            dy_sign = 0
            if uav.position.y < grid_center_y - 50:
                dy_sign = 1
            elif uav.position.y > grid_center_y + 50:
                dy_sign = -1

            dz_sign = 0
            # Maintain altitude (slight adjustment if too low)
            if uav.position.z < self.config.min_altitude + 10:
                dz_sign = 1

            # Add small randomness
            if random.random() < 0.2:
                dx_sign = random.choice([-1, 0, 1])
                dy_sign = random.choice([-1, 0, 1])
                dz_sign = random.choice([-1, 0, 0, 0])  # Bias toward no vertical movement

            direction = Direction3D(dx=dx_sign, dy=dy_sign, dz=dz_sign)
            directions.append(direction)

        return directions

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self._time_slot = 0

    @property
    def current_time_slot(self) -> int:
        """Get current time slot."""
        return self._time_slot
