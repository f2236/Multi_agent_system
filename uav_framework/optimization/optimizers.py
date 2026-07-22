"""Simple PSO optimizer for UAV positions (flattened vector)."""
from typing import Sequence, Tuple, List
import numpy as np
from copy import deepcopy
from uav_framework.simulation.uav import UAV


class PSOOptimizer:
    def __init__(self, evaluator, bounds: Tuple[Tuple[float, float], ...] = None):
        self.evaluator = evaluator
        self.bounds = bounds

    def _clip(self, vec: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return vec
        clipped = vec.copy()
        for i, (lo, hi) in enumerate(self.bounds):
            clipped[i] = float(np.clip(clipped[i], lo, hi))
        return clipped

    def optimize(self, uavs: Sequence[UAV], antennas: Sequence, iterations: int = 20, swarm_size: int = 16) -> Tuple[List[Tuple[float, float, float]], float]:
        n_uav = len(uavs)
        dim = n_uav * 3

        # initial flattened vector
        x0 = np.concatenate([np.array(u.pos, dtype=float) for u in uavs])

        # initialize swarm
        swarm = np.array([x0 + 0.1 * np.random.randn(dim) for _ in range(swarm_size)])
        velocities = np.zeros_like(swarm)

        pbest = swarm.copy()
        pbest_scores = np.full(swarm_size, -np.inf)

        # evaluate initial pbest
        for i in range(swarm_size):
            coords = swarm[i]
            objs = self._coords_to_uavs(coords, uavs)
            score = self.evaluator.evaluate(objs, antennas)
            pbest_scores[i] = score

        gbest_idx = int(np.argmax(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        w = 0.5
        c1 = 1.0
        c2 = 1.5

        for it in range(iterations):
            for i in range(swarm_size):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - swarm[i]) + c2 * r2 * (gbest - swarm[i])
                swarm[i] += velocities[i]
                swarm[i] = self._clip(swarm[i])

                objs = self._coords_to_uavs(swarm[i], uavs)
                score = self.evaluator.evaluate(objs, antennas)
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = swarm[i].copy()
                    if score > gbest_score:
                        gbest_score = score
                        gbest = swarm[i].copy()

        best_uavs = self._coords_to_uavs(gbest, uavs)
        best_positions = [tuple(u.pos) for u in best_uavs]
        return best_positions, float(gbest_score)

    def _coords_to_uavs(self, coords: np.ndarray, template_uavs: Sequence[UAV]) -> List[UAV]:
        n = len(template_uavs)
        uavs = []
        for i in range(n):
            base = template_uavs[i]
            x = float(coords[3 * i + 0])
            y = float(coords[3 * i + 1])
            z = float(coords[3 * i + 2])
            u = deepcopy(base)
            u.pos = (x, y, z)
            uavs.append(u)
        return uavs
