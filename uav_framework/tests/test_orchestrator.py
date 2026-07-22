from uav_framework.simulation.types import SimulationConfig
from uav_framework.simulation.environment import generate_uavs, generate_antennas
from uav_framework.optimization.objective import ObjectiveEvaluator
from uav_framework.agents.orchestrator import Orchestrator
from uav_framework.agents.evaluator_agent import EvaluatorAgent
from uav_framework.agents.planner import RandomPlanner
from uav_framework.simulation.uav import UAV


class FailingPlanner:
    def plan(self, uavs, antennas, context=None):
        raise RuntimeError("simulated planner outage")


def test_orchestrator_runs():
    cfg = SimulationConfig()
    uavs = generate_uavs(2, cfg)
    antennas = generate_antennas(1, cfg)

    evaluator = ObjectiveEvaluator()
    ev_agent = EvaluatorAgent(evaluator)

    orch = Orchestrator(cfg, planner=RandomPlanner(cfg))
    orch.set_evaluator(ev_agent)

    history = orch.run(uavs, antennas, iterations=3)
    assert len(history) == 3
    assert all(isinstance(s, float) for s in history)


def test_orchestrator_uses_fallback_when_planner_fails():
    cfg = SimulationConfig()
    uavs = [UAV(uav_id=0, pos=(100.0, 100.0, 50.0))]
    antennas = generate_antennas(1, cfg)

    evaluator = ObjectiveEvaluator()
    ev_agent = EvaluatorAgent(evaluator)

    orch = Orchestrator(cfg, planner=FailingPlanner())
    orch.set_evaluator(ev_agent)

    result = orch.run_iteration(uavs, antennas, iteration=1)

    assert result["planning_source"] == "Local best-candidate fallback"
    assert "simulated planner outage" in result["planner_error"]
    assert len(result["directions"]) == len(uavs)
    assert result["score"] >= result["old_rate"]
