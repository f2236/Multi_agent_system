"""Plotting utilities for UAV trajectories and convergence."""

from typing import List
import os

from uav_optimizer.domain import OptimizationState, UAVState


def save_trajectory_plot(state: OptimizationState, path: str) -> str:
    """Save a 3D trajectory plot for the best UAV positions.

    Args:
        state: OptimizationState after optimization
        path: output file path (PNG)

    Returns:
        The path written
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required to save plots") from e

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    uavs: List[UAVState] = state.best_uav_states
    for u in uavs:
        x, y, z = u.position.x, u.position.y, u.position.z
        ax.scatter([x], [y], [z], label=f"UAV {u.uav_id}")
        ax.text(x, y, z, f"{u.uav_id}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Best UAV Positions (Stable Stage)")
    ax.legend()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_convergence_plot(state: OptimizationState, path: str) -> str:
    """Save convergence plot (objective vs iteration).

    Args:
        state: OptimizationState with objective_history
        path: output file path (PNG)

    Returns:
        The path written
    """
    history = getattr(state, "objective_history", None)
    if not history:
        raise ValueError("No objective_history available in state to plot")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to save plots") from e

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(history)), history, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (R_S, bps)")
    ax.set_title("Convergence: Objective vs Iteration")
    ax.grid(True)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
