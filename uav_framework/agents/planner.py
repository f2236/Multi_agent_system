"""Planner implementations for UAV movement decisions."""

from dataclasses import dataclass, field
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from uav_framework.simulation.types import SimulationConfig


Direction = Tuple[int, int, int]


class PlanningError(RuntimeError):
    """Raised when the planner cannot produce a validated movement plan."""


@dataclass
class PlanningResult:
    """Validated movement plan and audit data for one iteration."""

    directions: List[Direction]
    prompt: str = ""
    response_text: str = ""
    selected_candidate_id: str = ""
    logs: List[str] = field(default_factory=list)


class RandomPlanner:
    """Generates simple direction vectors for each UAV.

    Directions are integer triplets in {-1,0,1} per axis.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def plan(
        self,
        uavs: List[Any],
        antennas: List[dict],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Direction]:
        dirs = []
        # Compute centroid of antennas as a simple attractor.
        if antennas:
            pts: List[Tuple[float, float, float]] = []
            for a in antennas:
                if isinstance(a, dict):
                    pos = a.get('pos')
                else:
                    pos = getattr(a, 'pos', None)

                pts.append(_position_tuple(pos))

            centroid = tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))
        else:
            centroid = None

        for u in uavs:
            if centroid is None:
                d = tuple(random.choice([-1, 0, 1]) for _ in range(3))
            else:
                vec = tuple(centroid[i] - float(u.pos[i]) for i in range(3))
                d = tuple(_sign(v) for v in vec)
            dirs.append(d)

        return dirs


class GroqPlanner:
    """Planner that calls Groq's OpenAI-compatible chat completions API.

    One call is made per optimization iteration. The planner only returns
    movement directions in {-1, 0, 1}; the simulator and objective evaluator
    remain local.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 30.0,
    ):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        self.config = config or SimulationConfig()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_base = (
            api_base or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        ).rstrip("/")
        self.timeout = timeout
        self.request_count = 0

    def plan(
        self,
        uavs: List[Any],
        antennas: List[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PlanningResult:
        """Call Groq and return a validated movement direction per UAV."""
        context = context or {}
        if not uavs:
            return PlanningResult(directions=[], logs=["Suggested movement: []"])

        prompt = self._build_prompt(uavs, antennas, context)
        logs = ["Calling Groq...", "Prompt sent."]
        response_text = self._call_groq(prompt)
        logs.append("Response received.")

        directions, selected_candidate_id = self._parse_directions(
            response_text,
            uavs,
            context.get("candidate_moves", []),
        )
        if selected_candidate_id:
            logs.append(f"Selected candidate: {selected_candidate_id}")
        logs.append(f"Suggested movement: {directions}")
        return PlanningResult(
            directions=directions,
            prompt=prompt,
            response_text=response_text,
            selected_candidate_id=selected_candidate_id,
            logs=logs,
        )

    def _call_groq(self, prompt: str) -> str:
        if not self.api_key:
            raise PlanningError("GROQ_API_KEY is not set, so Groq cannot be called.")

        try:
            import requests
        except Exception as exc:
            raise PlanningError("The requests package is required for Groq calls.") from exc

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Planner Agent in a multi-agent UAV optimization "
                        "framework. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            self.request_count += 1
        except Exception as exc:
            raise PlanningError(f"Groq request failed: {exc}") from exc

        if response.status_code >= 400:
            body = response.text[:800]
            raise PlanningError(f"Groq API error {response.status_code}: {body}")

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise PlanningError(f"Unexpected Groq response format: {response.text[:800]}") from exc

    def _build_prompt(
        self,
        uavs: List[Any],
        antennas: List[Any],
        context: Dict[str, Any],
    ) -> str:
        uav_payload = [
            {
                "uav_id": int(getattr(u, "uav_id", idx)),
                "x": float(u.pos[0]),
                "y": float(u.pos[1]),
                "z": float(u.pos[2]),
            }
            for idx, u in enumerate(uavs)
        ]
        receiver_payload = []
        for idx, antenna in enumerate(antennas):
            pos = antenna.get("pos") if isinstance(antenna, dict) else getattr(antenna, "pos", None)
            x, y, z = _position_tuple(pos)
            receiver_payload.append(
                {
                    "receiver_id": int(antenna.get("antenna_id", idx)) if isinstance(antenna, dict) else idx,
                    "fas_port_id": int(antenna.get("port_id", 1)) if isinstance(antenna, dict) else 1,
                    "K": int(antenna.get("K", 1)) if isinstance(antenna, dict) else 1,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

        state = {
            "iteration": context.get("iteration"),
            "current_stable_sum_data_rate_bps": context.get("current_rate"),
            "k_fixed": 1,
            "antenna_port_selection": "fixed to FAS port 1; do not optimize k",
            "uav_coordinates": uav_payload,
            "user_or_receiver_coordinates": context.get("users", receiver_payload),
            "single_fas_port_coordinate": receiver_payload[0] if receiver_payload else None,
            "precomputed_candidate_moves": context.get("candidate_moves", []),
            "constraints": {
                "movement_components_allowed": [-1, 0, 1],
                "uav_speed_mps": self.config.uav_speed,
                "time_step_seconds": self.config.time_step,
                "grid_x_bounds": list(self.config.grid_x_bounds),
                "grid_y_bounds": list(self.config.grid_y_bounds),
                "altitude_bounds": [self.config.min_altitude, self.config.max_altitude],
                "min_separation_m": self.config.min_separation,
                "max_separation_m": self.config.max_separation,
            },
        }

        return (
            "Maximize the stable-stage sum data rate objective R_S with K fixed to 1.\n"
            "The paper's k variable is the FAS antenna-port index; it is fixed "
            "to port 1 here, so do not select among multiple antennas or ports.\n"
            "Choose the next movement direction for every UAV. The local evaluator has "
            "already scored valid candidate moves below; use those scores as evidence.\n"
            "Each movement must be a direction vector with dx, dy, dz in {-1, 0, 1}.\n"
            "If any candidate has positive delta_rate_bps, you MUST choose a positive-delta "
            "candidate and MUST NOT return all-zero movement. Return only JSON in this shape:\n"
            '{"candidate_id":"C1","movements":[{"uav_id":0,"dx":0,"dy":0,"dz":0}],"rationale":"short reason"}\n\n'
            f"Current optimization state:\n{json.dumps(state, indent=2)}"
        )

    def _parse_directions(
        self,
        response_text: str,
        uavs: List[Any],
        candidate_moves: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Direction], str]:
        data = _extract_json_object(response_text)
        candidate_moves = candidate_moves or []

        candidate_id = str(data.get("candidate_id", "")).strip()
        if candidate_id:
            for candidate in candidate_moves:
                if candidate.get("candidate_id") == candidate_id:
                    return _candidate_directions(candidate, uavs), candidate_id

        movement_items = data.get("movements") or data.get("directions")
        if not isinstance(movement_items, list):
            raise PlanningError("Groq response did not contain a movements list.")

        by_id: Dict[int, Any] = {}
        ordered: List[Any] = []
        for item in movement_items:
            if isinstance(item, dict) and "uav_id" in item:
                by_id[int(item["uav_id"])] = item
            else:
                ordered.append(item)

        directions: List[Direction] = []
        for idx, uav in enumerate(uavs):
            uav_id = int(getattr(uav, "uav_id", idx))
            item = by_id.get(uav_id)
            if item is None and idx < len(ordered):
                item = ordered[idx]
            if item is None:
                raise PlanningError(f"Groq response missing movement for UAV {uav_id}.")
            directions.append(_coerce_direction(item))

        if _has_positive_nonzero_candidate(candidate_moves) and _all_zero(directions):
            raise PlanningError(
                "Groq returned all-zero movement even though the prompt contained "
                "positive-delta candidate moves."
            )

        matched = _matching_candidate_id(directions, candidate_moves)
        return directions, matched


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise PlanningError("Groq response did not include a JSON object.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Groq response JSON could not be parsed: {exc}") from exc


def _coerce_direction(item: Any) -> Direction:
    if isinstance(item, dict):
        raw = (item.get("dx"), item.get("dy"), item.get("dz"))
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        raw = (item[0], item[1], item[2])
    else:
        raise PlanningError(f"Invalid movement item: {item}")

    direction = tuple(_coerce_component(v) for v in raw)
    return direction  # type: ignore[return-value]


def _candidate_directions(candidate: Dict[str, Any], uavs: List[Any]) -> List[Direction]:
    raw = candidate.get("directions")
    if not isinstance(raw, list):
        raw = candidate.get("movements")
    if not isinstance(raw, list):
        raise PlanningError(f"Candidate {candidate.get('candidate_id')} has no directions.")

    by_id: Dict[int, Any] = {}
    ordered: List[Any] = []
    for item in raw:
        if isinstance(item, dict) and "uav_id" in item:
            by_id[int(item["uav_id"])] = item
        else:
            ordered.append(item)

    directions: List[Direction] = []
    for idx, uav in enumerate(uavs):
        uav_id = int(getattr(uav, "uav_id", idx))
        item = by_id.get(uav_id)
        if item is None and idx < len(ordered):
            item = ordered[idx]
        if item is None:
            raise PlanningError(f"Candidate missing movement for UAV {uav_id}.")
        directions.append(_coerce_direction(item))
    return directions


def _has_positive_nonzero_candidate(candidate_moves: List[Dict[str, Any]]) -> bool:
    for candidate in candidate_moves:
        directions = candidate.get("directions") or []
        if candidate.get("delta_rate_bps", 0.0) > 1e-9 and not _all_zero(directions):
            return True
    return False


def _matching_candidate_id(
    directions: List[Direction],
    candidate_moves: List[Dict[str, Any]],
) -> str:
    normalized = [tuple(d) for d in directions]
    for candidate in candidate_moves:
        candidate_dirs = [tuple(d) for d in candidate.get("directions", [])]
        if candidate_dirs == normalized:
            return str(candidate.get("candidate_id", ""))
    return ""


def _all_zero(directions: List[Any]) -> bool:
    return all(tuple(direction) == (0, 0, 0) for direction in directions)


def _coerce_component(value: Any) -> int:
    try:
        parsed = int(float(value))
    except Exception as exc:
        raise PlanningError(f"Movement component is not numeric: {value}") from exc

    if parsed not in (-1, 0, 1):
        raise PlanningError(f"Movement component {parsed} is outside {{-1, 0, 1}}.")
    return parsed


def _position_tuple(pos: Any) -> Tuple[float, float, float]:
    if hasattr(pos, "to_tuple"):
        pos = pos.to_tuple()
    if pos is None:
        raise PlanningError("Antenna or receiver is missing a position.")
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
