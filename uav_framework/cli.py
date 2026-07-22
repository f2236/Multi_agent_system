"""CLI entrypoint for headless Groq-planned runs."""
import argparse
import os
from uav_framework.simulation.types import SimulationConfig
from uav_framework.simulation.environment import generate_uavs, generate_antennas
from uav_framework.agents.orchestrator import Orchestrator
from uav_framework.agents.evaluator_agent import EvaluatorAgent
from uav_framework.optimization.objective import ObjectiveEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uavs', type=int, default=3)
    parser.add_argument('--antennas', type=int, default=1, choices=[1], help='Fixed K=1 single FAS port/receiver')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--export', type=str, default=None)
    args = parser.parse_args()

    cfg = SimulationConfig()
    uavs = generate_uavs(args.uavs, cfg)
    antennas = generate_antennas(args.antennas, cfg)

    orch = Orchestrator(cfg)
    evaluator = ObjectiveEvaluator()
    ev_agent = EvaluatorAgent(evaluator)
    orch.set_evaluator(ev_agent)

    history = orch.run(uavs, antennas, iterations=args.iterations)
    print('Final score', history[-1] if history else None)

    if args.export:
        from uav_framework.report.export import export_positions_csv
        os.makedirs(os.path.dirname(args.export) or '.', exist_ok=True)
        export_positions_csv(args.export, uavs, history)
        print('Exported to', args.export)


if __name__ == '__main__':
    main()
