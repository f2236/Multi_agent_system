"""Objective evaluator wrapper for the framework, reusing existing objective implementation."""
from typing import List, Sequence, Any
import numpy as np
from uav_framework.simulation.uav import UAV
import src.objective as core_objective


class ObjectiveEvaluator:
    def evaluate(self, uavs: Sequence[UAV], antennas: Sequence[Any], theta=None, K=1.0) -> float:
        if float(K) != 1.0:
            raise ValueError("Equation 23 is configured for K=1; antenna-port optimization is disabled.")
        if len(antennas) != 1:
            raise ValueError("The active paper/client setup requires exactly one FAS port/receiver for K=1.")

        # build uav_positions dict expected by src.objective.sum_data_rate
        uav_positions = {uav.uav_id: tuple(uav.pos) for uav in uavs}

        antenna_list = []
        for a in antennas:
            pos = None
            if isinstance(a, dict) and 'pos' in a:
                pos = a['pos']
            elif hasattr(a, 'pos'):
                pos = a.pos

            if pos is None:
                raise ValueError('antenna missing pos')

            if hasattr(pos, 'to_tuple'):
                antenna_list.append({'pos': pos.to_tuple()})
            elif isinstance(pos, (list, tuple)):
                antenna_list.append({'pos': tuple(pos)})
            else:
                # assume numpy array-like
                antenna_list.append({'pos': tuple(np.asarray(pos).tolist())})

        return core_objective.sum_data_rate(uav_positions, antenna_list, theta=theta, K=K)
