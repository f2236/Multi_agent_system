"""
Planner agent: proposes UAV position updates using Llama 3.

Responsibilities:
- Generate trajectory proposals
- Use Llama 3 / Groq for intelligent trajectory planning when configured
- Enforce movement constraints (Equation 20)
- Raise visible planning errors instead of silently masking LLM failures
"""

import json
import random
import math
from typing import List, Optional
from uav_optimizer.agents.base_agent import BaseAgent
from uav_optimizer.domain import OptimizationState, Position3D, Direction3D
from uav_optimizer.simulator.simulator import UAVSimulator


class PlannerAgent(BaseAgent):
    """Proposes UAV trajectory updates using Llama 3 or explicit heuristics."""

    def __init__(
        self,
        config,
        llama_interface=None,
    ):
        """
        Initialize planner agent.

        Args:
            config: Simulation configuration
            llama_interface: Optional Llama interface for LLM calls
        """
        super().__init__(name="PlannerAgent")
        self.config = config
        self.llama_interface = llama_interface
        self.simulator = UAVSimulator(config)

    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute trajectory planning.

        Args:
            state: Current optimization state

        Returns:
            Updated state with proposed positions
        """
        # Try to get proposals from Llama/Groq. If an interface is configured,
        # do not silently fall back on API or parsing failure.
        if self.llama_interface:
            if not self.llama_interface.is_available():
                raise RuntimeError("Configured Llama/Groq interface is not available")
            proposed = self._propose_via_llama(state)
            if proposed:
                state.proposed_positions = proposed
                self.log_execution(state, f"LLM proposed {len(proposed)} new positions")
                return state

        # Fallback to heuristic planner
        proposed = self._propose_heuristic(state)
        state.proposed_positions = proposed
        self.log_execution(
            state, f"Heuristic proposed {len(proposed)} new positions"
        )
        return state

    def _propose_via_llama(self, state: OptimizationState) -> Optional[List[Position3D]]:
        """
        Propose positions using Llama 3.

        Args:
            state: Current optimization state

        Returns:
            List of proposed positions or None if LLM failed
        """
        try:
            # Build prompt with current state
            prompt = self._build_planning_prompt(state)

            # Call Llama
            response = self.llama_interface.query(prompt)

            # Parse response
            positions = self._parse_llama_response(response, state)
            if positions is None:
                raise ValueError("LLM response did not contain valid proposed positions")
            return positions

        except Exception as e:
            state.agent_messages.append(
                {"agent": "PlannerAgent", "error": f"LLM failed: {str(e)}"}
            )
            raise RuntimeError(f"LLM planning failed: {str(e)}") from e

    def _propose_heuristic(self, state: OptimizationState) -> List[Position3D]:
        """
        Propose positions using heuristic algorithm.

        Strategy:
        - Move toward best antenna (closest or by SNR)
        - Maintain separation constraints
        - Add random exploration

        Args:
            state: Current optimization state

        Returns:
            List of proposed positions
        """
        proposed = []

        for uav in state.current_uav_states:
            # Generate heuristic direction
            direction = self._generate_heuristic_direction(uav, state)

            # Apply movement to get new position
            new_pos = self._apply_movement_to_position(
                uav.position, direction
            )

            proposed.append(new_pos)

        return proposed

    def _generate_heuristic_direction(
        self,
        uav,
        state: OptimizationState,
    ) -> Direction3D:
        """
        Generate heuristic movement direction.

        Strategy: Move toward grid center with occasional random changes.

        Args:
            uav: UAV state
            state: Optimization state

        Returns:
            Direction vector
        """
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds
        grid_cx = (x_min + x_max) / 2
        grid_cy = (y_min + y_max) / 2

        # Compute direction toward grid center
        dx = 0
        if uav.position.x < grid_cx - 50:
            dx = 1
        elif uav.position.x > grid_cx + 50:
            dx = -1

        dy = 0
        if uav.position.y < grid_cy - 50:
            dy = 1
        elif uav.position.y > grid_cy + 50:
            dy = -1

        dz = 0
        if uav.position.z < self.config.min_altitude + 20:
            dz = 1

        # Occasional randomness (20% chance)
        if random.random() < 0.2:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            dz = random.choice([-1, 0, 0, 0])

        return Direction3D(dx=dx, dy=dy, dz=dz)

    def _apply_movement_to_position(
        self, position: Position3D, direction: Direction3D
    ) -> Position3D:
        """
        Apply movement model to compute new position.

        Equation 20: new_pos = old_pos + V*Δt*direction_unit

        Args:
            position: Current position
            direction: Movement direction

        Returns:
            New position
        """
        step_magnitude = self.config.uav_speed * self.config.time_step_duration

        # Direction as unit vector
        direction_vec = [float(direction.dx), float(direction.dy), float(direction.dz)]
        norm = math.sqrt(sum(x**2 for x in direction_vec))

        if norm < 1e-6:
            return position

        direction_unit = [x / norm for x in direction_vec]
        step_vec = [step_magnitude * x for x in direction_unit]

        # Compute new position
        new_x = position.x + step_vec[0]
        new_y = position.y + step_vec[1]
        new_z = position.z + step_vec[2]

        # Enforce bounds
        new_z = max(self.config.min_altitude, min(new_z, self.config.max_altitude))
        x_min, x_max = self.config.grid_x_bounds
        y_min, y_max = self.config.grid_y_bounds
        new_x = max(x_min, min(new_x, x_max))
        new_y = max(y_min, min(new_y, y_max))

        return Position3D(x=new_x, y=new_y, z=new_z)

    def _build_planning_prompt(self, state: OptimizationState) -> str:
        """
        Build prompt for Llama 3 trajectory planning.

        Args:
            state: Current optimization state

        Returns:
            Prompt string
        """
        uav_info = []
        for uav in state.current_uav_states:
            uav_info.append(
                f"UAV {uav.uav_id}: position ({uav.position.x:.1f}, {uav.position.y:.1f}, {uav.position.z:.1f})"
            )

        antenna_info = []
        for antenna in state.antenna_configs:
            antenna_info.append(
                f"Antenna {antenna.antenna_id}: ({antenna.position.x:.1f}, {antenna.position.y:.1f}, {antenna.position.z:.1f})"
            )

        prompt = (
            f"You are a UAV trajectory optimization agent.\n"
            f"Current state (iteration {state.iteration}):\n"
            f"  Best objective so far: {state.best_objective:.2e}\n"
            f"\n"
            f"Current UAV positions:\n"
            + "\n".join(uav_info)
            + f"\n\nAntenna positions:\n"
            + "\n".join(antenna_info)
            + f"\n\nConstraints:\n"
            f"  - Max speed: {self.config.uav_speed} m/s\n"
            f"  - Min UAV separation: {self.config.min_uav_separation} m\n"
            f"  - Altitude bounds: [{self.config.min_altitude}, {self.config.max_altitude}] m\n"
            f"\nPropose next positions for each UAV as JSON:\n"
            f'{{"positions": [{{"uav_id": 0, "x": ..., "y": ..., "z": ...}}, ...]}}\n'
        )
        return prompt

    def _parse_llama_response(
        self, response: str, state: OptimizationState
    ) -> Optional[List[Position3D]]:
        """
        Parse Llama response and extract positions.

        Args:
            response: LLM response text
            state: Optimization state

        Returns:
            List of Position3D or None if parsing failed
        """
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                positions = []
                for pos_data in data.get("positions", []):
                    pos = Position3D(
                        x=float(pos_data["x"]),
                        y=float(pos_data["y"]),
                        z=float(pos_data["z"]),
                    )
                    positions.append(pos)

                if len(positions) == len(state.current_uav_states):
                    return positions

        except Exception:
            pass

        return None
