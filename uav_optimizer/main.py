"""
Main entry point for UAV trajectory optimization framework.

Demonstrates complete multi-agent optimization loop.
"""

import sys
import random

# Import domain FIRST before numpy to avoid types.py conflict
from uav_optimizer.domain import (
    Position3D,
    UAVState,
    SimulationConfig,
    AntennaConfig,
)

import numpy as np
from uav_optimizer.config import Config
from uav_optimizer.agents.orchestrator import OptimizationOrchestrator
from uav_optimizer.llm.llama import LlamaInterface
from uav_optimizer.visualization.plots import save_trajectory_plot, save_convergence_plot
import os


def initialize_scenario(
    num_uavs: int = 3,
    config: SimulationConfig = None,
) -> tuple:
    """
    Initialize UAVs and antennas for optimization scenario.

    Args:
        num_uavs: Number of UAVs
        config: Simulation configuration

    Returns:
        (initial_uav_states, antennas, config)
    """
    if config is None:
        config = SimulationConfig()

    # Random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Initialize UAVs at random positions
    uav_states = []
    x_min, x_max = config.grid_x_bounds
    y_min, y_max = config.grid_y_bounds

    for i in range(num_uavs):
        pos = Position3D(
            x=random.uniform(x_min, x_max),
            y=random.uniform(y_min, y_max),
            z=random.uniform(config.min_altitude + 10, config.min_altitude + 100),
        )
        uav = UAVState(uav_id=i, position=pos, antenna_port=1)
        uav_states.append(uav)

    # Initialize antennas at grid center
    antennas = []
    for i in range(config.num_antennas):
        ant_pos = Position3D(
            x=(x_min + x_max) / 2,
            y=(y_min + y_max) / 2,
            z=20.0,
        )
        antenna = AntennaConfig(
            antenna_id=i,
            position=ant_pos,
            tx_power_dbm=20.0,
            bandwidth_hz=1e6,
            frequency_hz=2.4e9,
            noise_figure_db=7.0,
        )
        antennas.append(antenna)

    return uav_states, antennas, config


def main():
    """Run UAV trajectory optimization."""
    print("UAV Trajectory Optimization - Research Framework")
    print("=" * 70)

    # Load configuration
    cfg = Config()
    config = cfg.simulation

    print(f"Configuration loaded:")
    print(f"  - Time steps: {config.num_steps}")
    print(f"  - Stable stage: {config.stable_stage_slot}")
    print(f"  - UAV speed: {config.uav_speed} m/s")
    print(f"  - Grid bounds: {config.grid_x_bounds} x {config.grid_y_bounds}")
    print()

    # Initialize scenario
    num_uavs = 3
    uav_states, antennas, config = initialize_scenario(num_uavs, config)

    print(f"Scenario initialized:")
    print(f"  - UAVs: {len(uav_states)}")
    print(f"  - Antennas: {len(antennas)}")
    print(f"  - Grid size: {config.grid_x_bounds[1] - config.grid_x_bounds[0]:.0f} x "
          f"{config.grid_y_bounds[1] - config.grid_y_bounds[0]:.0f} m")
    print()

    # Initialize Llama/Groq interface. A real run should not silently switch to
    # heuristic movement when the API is missing.
    llama_interface = LlamaInterface()
    if not llama_interface.is_available():
        raise RuntimeError(
            "Llama/Groq interface is not available. Set GROQ_API_KEY for Groq "
            "or explicitly configure another provider."
        )
    print(f"Llama 3 interface available: {llama_interface}")

    print()

    # Create orchestrator
    orchestrator = OptimizationOrchestrator(
        config=config,
        antennas=antennas,
        initial_uav_states=uav_states,
        llama_interface=llama_interface,
    )

    # Run orchestrator optimization (agent loop)
    final_state = orchestrator.run(verbose=True)

    # Print results
    summary = orchestrator.get_optimization_summary()
    print("\nOptimization Results:")
    print(f"  - Best objective (R_S): {summary['best_objective']:.2e} bps")
    print(f"  - Iterations: {summary['iterations']}")
    print(f"  - Convergence: {summary['convergence_reached']}")
    print()

    print("Best UAV Positions:")
    for uav_id, pos in summary["best_positions"]:
        print(f"  UAV {uav_id}: {pos}")

    # Save visualizations (if available)
    try:
        out_dir = os.path.join("outputs", "plots")
        os.makedirs(out_dir, exist_ok=True)
        traj_path = os.path.join(out_dir, "best_positions.png")
        conv_path = os.path.join(out_dir, "convergence.png")
        save_trajectory_plot(final_state, traj_path)
        save_convergence_plot(final_state, conv_path)
        print(f"Saved plots to: {out_dir}")
    except Exception as e:
        print(f"Could not save plots: {e}")

    return final_state


if __name__ == "__main__":
    try:
        final_state = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
