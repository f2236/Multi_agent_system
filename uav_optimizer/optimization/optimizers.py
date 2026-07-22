"""Advanced optimizers for UAV trajectory optimization.

Includes a Particle Swarm Optimization (PSO) implementation that optimizes
final UAV positions (stable-stage) to maximize Equation 22 objective.
"""
from typing import List, Tuple, Optional
import numpy as np
from uav_optimizer.domain import UAVState, Position3D, SimulationConfig
from uav_optimizer.optimization.objective import ObjectiveEvaluator
from uav_optimizer.simulator.constraints import ConstraintValidator


class PSOOptimizer:
    """Simple Particle Swarm Optimizer for continuous UAV final positions.

    The optimizer searches over the 3D positions of all UAVs at the stable
    stage (T+2). It enforces bounds and relies on the ObjectiveEvaluator to
    return a zero objective for invalid states (constraints violated).
    """

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        antennas: List,
        config: SimulationConfig,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: Optional[int] = None,
    ):
        self.evaluator = evaluator
        self.antennas = antennas
        self.config = config
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        if seed is not None:
            np.random.seed(seed)

    def _flatten(self, uav_states: List[UAVState]) -> np.ndarray:
        return np.array([coord for u in uav_states for coord in (u.position.x, u.position.y, u.position.z)])

    def _unflatten(self, vec: np.ndarray) -> List[UAVState]:
        vals = vec.reshape(-1, 3)
        return [UAVState(uav_id=i, position=Position3D(float(x), float(y), float(z)), antenna_port=1) for i, (x, y, z) in enumerate(vals)]

    def _clip(self, vec: np.ndarray) -> np.ndarray:
        vals = vec.reshape(-1, 3)
        clipped = []
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds
        z_min, z_max = self.config.min_altitude, self.config.max_altitude
        for x, y, z in vals:
            cx = float(np.clip(x, x_min, x_max))
            cy = float(np.clip(y, y_min, y_max))
            cz = float(np.clip(z, z_min, z_max))
            clipped.append((cx, cy, cz))
        return np.array(clipped).reshape(-1)

    def optimize(self, initial_states: List[UAVState]) -> Tuple[List[UAVState], List[float]]:
        """Run PSO to optimize final UAV positions.

        Returns:
            best_states: List[UAVState] at best found objective
            history: List of best objective per iteration
        """
        dim = len(initial_states) * 3

        # Initialize particles around the initial states
        x0 = self._flatten(initial_states)
        spread = np.maximum(1.0, np.array([self.config.grid_x_bounds[1] - self.config.grid_x_bounds[0],
                                           self.config.grid_y_bounds[1] - self.config.grid_y_bounds[0],
                                           max(1.0, self.config.max_altitude - self.config.min_altitude)])).repeat(len(initial_states))

        particles = np.tile(x0, (self.num_particles, 1)) + (np.random.randn(self.num_particles, dim) * (spread * 0.05))
        velocities = np.zeros_like(particles)

        pbest = particles.copy()
        pbest_val = np.full(self.num_particles, -np.inf)

        gbest = particles[0].copy()
        gbest_val = -np.inf

        history = []

        for it in range(self.max_iter):
            for i in range(self.num_particles):
                # Clip particle into bounds
                pos = self._clip(particles[i])
                candidate_states = self._unflatten(pos)

                # Evaluate
                res = self.evaluator.evaluate(candidate_states, self.antennas, iteration=it)
                fitness = res.value if res.constraints_satisfied else 0.0

                # Update personal best
                if fitness > pbest_val[i]:
                    pbest_val[i] = fitness
                    pbest[i] = particles[i].copy()

                # Update global best
                if fitness > gbest_val:
                    gbest_val = fitness
                    gbest = particles[i].copy()

            history.append(gbest_val)

            # Update velocities and positions
            r1 = np.random.rand(self.num_particles, dim)
            r2 = np.random.rand(self.num_particles, dim)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - particles)
                + self.c2 * r2 * (gbest - particles)
            )

            particles += velocities

            # Small jitter to avoid stagnation
            particles += 0.01 * np.random.randn(*particles.shape)

            # Optional early stopping when improvement tiny
            if it > 5 and abs(history[-1] - history[-2]) < (self.config.convergence_threshold * 1e-2):
                break

        best_states = self._unflatten(self._clip(gbest))
        return best_states, history
