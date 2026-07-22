from copy import deepcopy
from typing import List, Any
import math
from uav_framework.simulation.uav import UAV
from uav_framework.simulation.simulator import UAVSimulator
from uav_framework.agents.planner import GroqPlanner, PlanningResult
from uav_framework.agents.environment_agent import EnvironmentAgent
from uav_framework.agents.evaluator_agent import EvaluatorAgent
from uav_framework.simulation.types import SimulationConfig


class Orchestrator:
    def __init__(self, config: SimulationConfig, planner=None):
        self.config = config
        self.sim = UAVSimulator(config)
        self.planner = planner or GroqPlanner(config)
        self.evaluator = EvaluatorAgent(None)  # to be set externally
        self.history = []
        self.last_iteration_logs = []

    def set_evaluator(self, evaluator_agent: EvaluatorAgent):
        self.evaluator = evaluator_agent

    def run_iteration(self, uavs: List[UAV], antennas: List[Any], iteration: int = 1):
        """Run one audited optimization iteration.

        Movement decisions come only from the configured planner. The simulator
        and objective evaluator remain local.
        """
        old_positions = [tuple(u.pos) for u in uavs]
        old_rate = self.evaluator.evaluate(uavs, antennas)
        candidate_moves = self._build_candidate_moves(uavs, antennas, old_rate)
        best_candidate = candidate_moves[0] if candidate_moves else None
        logs = [f"Iteration: {iteration}"]
        if best_candidate:
            logs.append(
                "Best candidate before Groq: "
                f"{best_candidate['candidate_id']} delta={best_candidate['delta_rate_bps']:.6f}"
            )

        context = {
            "iteration": iteration,
            "current_rate": old_rate,
            "candidate_moves": candidate_moves,
            "constraints": {
                "time_step": self.config.time_step,
                "uav_speed": self.config.uav_speed,
                "grid_x_bounds": self.config.grid_x_bounds,
                "grid_y_bounds": self.config.grid_y_bounds,
                "min_altitude": self.config.min_altitude,
                "max_altitude": self.config.max_altitude,
                "min_separation": self.config.min_separation,
                "max_separation": self.config.max_separation,
            },
        }

        planning_source = (
            "Groq Llama 3"
            if isinstance(self.planner, GroqPlanner)
            else self.planner.__class__.__name__
        )
        planner_error = ""

        try:
            plan = self.planner.plan(uavs, antennas, context=context)
            if isinstance(plan, PlanningResult):
                directions = plan.directions
                logs.extend(plan.logs)
                prompt = plan.prompt
                response_text = plan.response_text
                selected_candidate_id = plan.selected_candidate_id
            else:
                directions = plan
                logs.append(f"Suggested movement: {directions}")
                prompt = ""
                response_text = ""
                selected_candidate_id = ""
        except Exception as exc:
            planner_error = str(exc)
            fallback = self._select_fallback_candidate(candidate_moves)
            directions = self._directions_from_candidate(fallback)
            if len(directions) != len(uavs):
                directions = [(0, 0, 0) for _ in uavs]
            selected_candidate_id = str(fallback.get("candidate_id", "LOCAL_FALLBACK"))
            prompt = ""
            response_text = (
                "Local continuity fallback used because the Groq planner was "
                f"unavailable or returned invalid output: {planner_error}"
            )
            planning_source = "Local best-candidate fallback"
            logs.extend(
                [
                    f"Planner continuity mode: {planning_source}",
                    f"Fallback candidate: {selected_candidate_id}",
                    f"Fallback reason: {planner_error}",
                    f"Suggested movement: {directions}",
                ]
            )

        self.sim.step(uavs, directions)
        suggested_positions = [tuple(u.pos) for u in uavs]
        constraints_valid, violations = self._validate_constraints(uavs)
        new_rate = self.evaluator.evaluate(uavs, antennas) if constraints_valid else 0.0

        accepted = constraints_valid and new_rate >= old_rate
        if not accepted:
            for uav, old_pos in zip(uavs, old_positions):
                uav.pos = old_pos

        current_rate = new_rate if accepted else old_rate
        self.history.append(current_rate)
        decision = self._decision_label(directions, old_rate, new_rate, accepted)

        logs.extend(
            [
                f"New UAV Position: {suggested_positions}",
                f"Old Rate: {old_rate:.6f}",
                f"New Rate: {new_rate:.6f}",
                f"Constraints valid: {constraints_valid}",
                f"Accepted/Rejected: {decision}",
            ]
        )
        if violations:
            logs.extend([f"Constraint violation: {v}" for v in violations[:3]])
        self.last_iteration_logs = logs
        self._print_logs(logs)

        return {
            "iteration": iteration,
            "directions": directions,
            "selected_candidate_id": selected_candidate_id,
            "candidate_moves": candidate_moves,
            "old_positions": old_positions,
            "suggested_positions": suggested_positions,
            "current_positions": [tuple(u.pos) for u in uavs],
            "old_rate": old_rate,
            "new_rate": new_rate,
            "score": current_rate,
            "accepted": accepted,
            "decision": decision,
            "constraints_valid": constraints_valid,
            "constraint_violations": violations,
            "logs": logs,
            "prompt": prompt,
            "response_text": response_text,
            "planning_source": planning_source,
            "planner_error": planner_error,
        }

    def run(self, uavs: List[UAV], antennas: List[Any], iterations: int = 10):
        history = []
        for it in range(iterations):
            result = self.run_iteration(uavs, antennas, iteration=it + 1)
            history.append(result["score"])
        return history

    @staticmethod
    def _print_logs(logs):
        for line in logs:
            print(line)

    def _build_candidate_moves(self, uavs: List[UAV], antennas: List[Any], old_rate: float):
        """Precompute locally-evaluated candidate moves for the Groq planner."""
        directions = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
        zero = [(0, 0, 0) for _ in uavs]
        candidates = []

        hold_score, hold_positions, hold_valid, hold_violations = self._score_candidate(
            uavs,
            antennas,
            zero,
        )
        candidates.append(
            self._candidate_payload(
                "HOLD",
                zero,
                hold_score,
                old_rate,
                hold_positions,
                hold_valid,
                hold_violations,
            )
        )

        best_single_by_uav = []
        for uav_idx, _uav in enumerate(uavs):
            scored = []
            for direction in directions:
                if direction == (0, 0, 0):
                    continue
                proposal = list(zero)
                proposal[uav_idx] = direction
                score, positions, valid, violations = self._score_candidate(
                    uavs, antennas, proposal
                )
                if valid:
                    scored.append((score, proposal, positions, valid, violations))

            scored.sort(key=lambda item: item[0], reverse=True)
            if scored:
                best_single_by_uav.append(scored[0])
                for rank, (score, proposal, positions, valid, violations) in enumerate(
                    scored[:2],
                    start=1,
                ):
                    candidates.append(
                        self._candidate_payload(
                            f"U{uav_idx}_R{rank}",
                            proposal,
                            score,
                            old_rate,
                            positions,
                            valid,
                            violations,
                        )
                    )

        combined = list(zero)
        for score, proposal, _positions, _valid, _violations in best_single_by_uav:
            if score > old_rate:
                for idx, direction in enumerate(proposal):
                    if direction != (0, 0, 0):
                        combined[idx] = direction
        if any(direction != (0, 0, 0) for direction in combined):
            score, positions, valid, violations = self._score_candidate(
                uavs,
                antennas,
                combined,
            )
            if valid:
                candidates.append(
                    self._candidate_payload(
                        "COMBINED_BEST",
                        combined,
                        score,
                        old_rate,
                        positions,
                        valid,
                        violations,
                    )
                )

        unique = {}
        for candidate in candidates:
            key = tuple(tuple(direction) for direction in candidate["directions"])
            if key not in unique or candidate["predicted_rate_bps"] > unique[key]["predicted_rate_bps"]:
                unique[key] = candidate

        sorted_candidates = sorted(
            unique.values(),
            key=lambda item: item["predicted_rate_bps"],
            reverse=True,
        )
        return sorted_candidates[:8]

    def _score_candidate(self, uavs: List[UAV], antennas: List[Any], directions):
        candidate_uavs = deepcopy(uavs)
        self.sim.step(candidate_uavs, directions)
        valid, violations = self._validate_constraints(candidate_uavs)
        score = self.evaluator.evaluate(candidate_uavs, antennas) if valid else 0.0
        positions = [tuple(uav.pos) for uav in candidate_uavs]
        return score, positions, valid, violations

    @staticmethod
    def _candidate_payload(candidate_id, directions, score, old_rate, positions, valid, violations):
        return {
            "candidate_id": candidate_id,
            "directions": [tuple(direction) for direction in directions],
            "movements": [
                {"uav_id": idx, "dx": int(direction[0]), "dy": int(direction[1]), "dz": int(direction[2])}
                for idx, direction in enumerate(directions)
            ],
            "predicted_rate_bps": float(score),
            "delta_rate_bps": float(score - old_rate),
            "feasible": bool(valid),
            "constraint_violations": list(violations),
            "predicted_positions": [
                [float(pos[0]), float(pos[1]), float(pos[2])]
                for pos in positions
            ],
        }

    @staticmethod
    def _select_fallback_candidate(candidate_moves):
        """Choose the best safe local candidate when the external planner fails."""
        feasible = [candidate for candidate in candidate_moves if candidate.get("feasible", False)]
        positive = [
            candidate
            for candidate in feasible
            if candidate.get("delta_rate_bps", 0.0) > 1e-9
            and not all(tuple(direction) == (0, 0, 0) for direction in candidate.get("directions", []))
        ]
        if positive:
            return positive[0]
        if feasible:
            return feasible[0]
        if candidate_moves:
            return candidate_moves[0]
        return {
            "candidate_id": "LOCAL_HOLD",
            "directions": [],
            "predicted_rate_bps": 0.0,
            "delta_rate_bps": 0.0,
            "feasible": True,
        }

    @staticmethod
    def _directions_from_candidate(candidate):
        return [tuple(direction) for direction in candidate.get("directions", [])]

    @staticmethod
    def _decision_label(directions, old_rate, new_rate, accepted):
        if all(tuple(direction) == (0, 0, 0) for direction in directions):
            return "No-op"
        if accepted and new_rate > old_rate:
            return "Accepted"
        if accepted:
            return "Accepted"
        return "Rejected"

    def _validate_constraints(self, uavs: List[UAV]):
        violations = []
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds

        for uav in uavs:
            x, y, z = (float(v) for v in uav.pos)
            if not (x_min <= x <= x_max):
                violations.append(f"UAV {uav.uav_id} x={x:.2f} outside [{x_min}, {x_max}]")
            if not (y_min <= y <= y_max):
                violations.append(f"UAV {uav.uav_id} y={y:.2f} outside [{y_min}, {y_max}]")
            if not (self.config.min_altitude <= z <= self.config.max_altitude):
                violations.append(
                    f"UAV {uav.uav_id} z={z:.2f} outside "
                    f"[{self.config.min_altitude}, {self.config.max_altitude}]"
                )

        for i in range(len(uavs)):
            for j in range(i + 1, len(uavs)):
                distance = math.dist(uavs[i].pos, uavs[j].pos)
                if distance < self.config.min_separation:
                    violations.append(
                        f"UAV {uavs[i].uav_id} and UAV {uavs[j].uav_id} "
                        f"distance {distance:.2f} < d_min {self.config.min_separation}"
                    )

        return len(violations) == 0, violations
