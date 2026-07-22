from typing import List, Any
from uav_framework.simulation.uav import UAV
from uav_framework.optimization.objective import ObjectiveEvaluator


class EvaluatorAgent:
    def __init__(self, evaluator: ObjectiveEvaluator):
        self.evaluator = evaluator
        self.history = []

    def evaluate(self, uavs: List[UAV], antennas: List[Any]):
        score = self.evaluator.evaluate(uavs, antennas)
        self.history.append(score)
        return score
