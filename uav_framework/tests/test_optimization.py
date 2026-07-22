import pytest

from uav_framework.simulation.types import SimulationConfig
from uav_framework.simulation.environment import generate_uavs, generate_antennas
from uav_framework.optimization.objective import ObjectiveEvaluator
from uav_framework.optimization.optimizers import PSOOptimizer


def test_pso_improves_objective():
    cfg = SimulationConfig()
    uavs = generate_uavs(1, cfg)
    antennas = generate_antennas(1, cfg)

    evaluator = ObjectiveEvaluator()
    initial = evaluator.evaluate(uavs, antennas)

    pso = PSOOptimizer(evaluator)
    best_positions, best_score = pso.optimize(uavs, antennas, iterations=8, swarm_size=8)

    assert best_score >= initial


def test_k1_single_fas_port_is_enforced():
    cfg = SimulationConfig()
    uavs = generate_uavs(1, cfg)
    antennas = generate_antennas(1, cfg)

    assert len(antennas) == 1
    assert antennas[0]["K"] == 1
    assert antennas[0]["port_id"] == 1

    evaluator = ObjectiveEvaluator()
    assert evaluator.evaluate(uavs, antennas, K=1) > 0.0

    with pytest.raises(ValueError):
        generate_antennas(2, cfg)

    with pytest.raises(ValueError):
        evaluator.evaluate(uavs, antennas + antennas, K=1)

    with pytest.raises(ValueError):
        evaluator.evaluate(uavs, antennas, K=2)
