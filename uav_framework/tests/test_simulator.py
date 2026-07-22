from uav_framework.simulation.types import SimulationConfig
from uav_framework.simulation.environment import generate_uavs
from uav_framework.simulation.simulator import UAVSimulator


def test_step_moves_uav():
    cfg = SimulationConfig()
    uavs = generate_uavs(1, cfg)
    sim = UAVSimulator(cfg)
    before = uavs[0].pos
    # move east (dx=1)
    sim.step(uavs, [(1,0,0)])
    after = uavs[0].pos
    assert after != before
