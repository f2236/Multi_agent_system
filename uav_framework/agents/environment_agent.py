from typing import List, Tuple
from uav_framework.simulation.uav import UAV
from uav_framework.simulation.simulator import UAVSimulator


class EnvironmentAgent:
    def __init__(self, sim: UAVSimulator):
        self.sim = sim

    def apply(self, uavs: List[UAV], directions: List[Tuple[int, int, int]]) -> List[UAV]:
        return self.sim.step(uavs, directions)
