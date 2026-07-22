"""Compatibility entry point for Groq-planned UAV optimization."""

import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_framework.agents.evaluator_agent import EvaluatorAgent
from uav_framework.agents.orchestrator import Orchestrator
from uav_framework.agents.planner import GroqPlanner
from uav_framework.optimization.objective import ObjectiveEvaluator
from uav_framework.simulation.environment import generate_antennas, generate_uavs
from uav_framework.simulation.types import SimulationConfig


def run_example(num_uavs=3, iters=20):
    random.seed(0)

    cfg = SimulationConfig()
    uavs = generate_uavs(num_uavs, cfg)
    antennas = generate_antennas(1, cfg)

    orchestrator = Orchestrator(cfg, planner=GroqPlanner(cfg))
    orchestrator.set_evaluator(EvaluatorAgent(ObjectiveEvaluator()))

    print("Planner: Groq Llama 3 chat completions")
    history = orchestrator.run(uavs, antennas, iterations=iters)

    print("Final UAV positions:")
    for uav in uavs:
        print(uav.uav_id, tuple(uav.pos))

    if history:
        print(f"Final stable sum data rate: {history[-1]:.3f}")


if __name__ == "__main__":
    run_example()
