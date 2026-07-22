"""Environment generation: users, single FAS port, initial UAV placement."""
from typing import List, Tuple
import random
from .types import Position3D, SimulationConfig
from .uav import UAV


def generate_antennas(n: int, config: SimulationConfig) -> List[dict]:
    """Generate the single communication/FAS port used by the K=1 setup.

    The paper's antenna variable k is a selectable FAS port index
    (Equation 23), not a request for multiple physical receiver objects. For
    the client scenario K is fixed to 1, so this generator intentionally
    returns exactly one fixed port/receiver.
    """
    if n != 1:
        raise ValueError("This project is configured for K=1, so exactly one FAS port/receiver is allowed.")

    x_min, x_max = config.grid_x_bounds
    y_min, y_max = config.grid_y_bounds
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0
    z = 20.0
    return [
        {
            "antenna_id": 0,
            "port_id": 1,
            "K": 1,
            "label": "FAS port 1 (K=1)",
            "pos": Position3D(x=x, y=y, z=z),
        }
    ]


def generate_uavs(n: int, config: SimulationConfig) -> List[UAV]:
    x_min, x_max = config.grid_x_bounds
    y_min, y_max = config.grid_y_bounds
    uavs: List[UAV] = []
    for i in range(n):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(config.min_altitude + 5, config.min_altitude + 100)
        uavs.append(UAV(uav_id=i, pos=(x, y, z)))
    return uavs
