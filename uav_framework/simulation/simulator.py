"""UAV simulator implementing Equation 20 movement model and constraints."""
from typing import List, Sequence, Tuple
import numpy as np
from loguru import logger
from .types import SimulationConfig, Position3D
from .uav import UAV


class UAVSimulator:
    """Simulates UAV movements per time step.

    Directions are expected as sequences of integers in {-1,0,1} per axis.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def step(self, uavs: List[UAV], directions: Sequence[Tuple[int, int, int]]) -> List[UAV]:
        """Apply one simulation step given directions for each UAV.

        Args:
            uavs: list of UAV objects
            directions: list of (dx,dy,dz) per UAV where each in {-1,0,1}

        Returns:
            Updated list of UAVs (in-place update applied)
        """
        if len(directions) != len(uavs):
            raise ValueError("directions must match number of UAVs")

        dt = self.config.time_step
        v = self.config.uav_speed

        for uav, dir_triplet in zip(uavs, directions):
            arr = uav.as_array()
            if any(int(component) not in (-1, 0, 1) for component in dir_triplet):
                raise ValueError("direction components must be in {-1, 0, 1}")

            # Equation 20: s_{n,t+1} = s_{n,t} + V * delta_t * [dx, dy, dz].
            # The paper applies the component vector directly; it is not
            # normalized for diagonal movement.
            dir_vec = np.array(
                [float(dir_triplet[0]), float(dir_triplet[1]), float(dir_triplet[2])]
            )
            step = dir_vec * v * dt

            new_pos = arr + step

            # enforce bounds
            x_min, x_max = self.config.grid_x_bounds
            y_min, y_max = self.config.grid_y_bounds
            new_pos[0] = float(np.clip(new_pos[0], x_min, x_max))
            new_pos[1] = float(np.clip(new_pos[1], y_min, y_max))
            new_pos[2] = float(np.clip(new_pos[2], self.config.min_altitude, self.config.max_altitude))

            # update UAV
            uav.update_from_array(new_pos)

        # optional: check separations and log violations
        self._log_separation_violations(uavs)

        return uavs

    def _log_separation_violations(self, uavs: List[UAV]) -> None:
        n = len(uavs)
        for i in range(n):
            for j in range(i + 1, n):
                di = np.linalg.norm(np.array(uavs[i].pos) - np.array(uavs[j].pos))
                if di < self.config.min_separation:
                    logger.warning(f"UAV {uavs[i].uav_id} and {uavs[j].uav_id} too close: {di:.2f} m")
