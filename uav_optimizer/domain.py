"""
Domain types and data structures for UAV trajectory optimization.

Maps directly to research paper equations and constraints.
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ObjectiveType(Enum):
    """Objective function type."""
    STABLE_STAGE_SUM_RATE = "R_S"  # Equation 22


@dataclass(frozen=True)
class Position3D:
    """3D position vector (x, y, z) in meters."""
    x: float
    y: float
    z: float

    def __iter__(self):
        """Allow unpacking as tuple."""
        return iter((self.x, self.y, self.z))

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Direction3D:
    """Movement direction vector: each element in {-1, 0, +1}."""
    dx: int
    dy: int
    dz: int

    def __post_init__(self):
        for val in (self.dx, self.dy, self.dz):
            if val not in (-1, 0, 1):
                raise ValueError(f"Direction component must be in {{-1, 0, +1}}, got {val}")


@dataclass(frozen=True)
class AntennaConfig:
    """Fixed antenna configuration."""
    antenna_id: int
    position: Position3D
    max_users: int = 10
    tx_power_dbm: float = 20.0
    bandwidth_hz: float = 1e6
    frequency_hz: float = 2.4e9
    noise_figure_db: float = 7.0


@dataclass
class UAVState:
    """UAV state at a given time slot."""
    uav_id: int
    position: Position3D
    antenna_port: int = 0  # Fixed antenna selection (equation 23)

    def distance_to(self, other: "UAVState") -> float:
        """Euclidean distance to another UAV."""
        dx = self.position.x - other.position.x
        dy = self.position.y - other.position.y
        dz = self.position.z - other.position.z
        return (dx**2 + dy**2 + dz**2) ** 0.5


@dataclass
class ChannelState:
    """Channel properties between UAV and antenna."""
    uav_id: int
    antenna_id: int
    distance: float
    path_loss_db: float
    snr_db: float
    data_rate_bps: float


@dataclass
class SimulationConfig:
    """Simulation and optimization configuration."""
    # Time slots
    time_step_duration: float = 1.0  # Δt in seconds
    num_steps: int = 10  # T value from paper
    stable_stage_slot: Optional[int] = None  # T+2, auto-computed if None

    # UAV dynamics (Equation 20)
    uav_speed: float = 10.0  # V in m/s
    min_altitude: float = 5.0
    max_altitude: float = 500.0
    grid_x_bounds: Tuple[float, float] = (0.0, 1000.0)
    grid_y_bounds: Tuple[float, float] = (0.0, 1000.0)

    # Constraints (Equation 24)
    min_uav_separation: float = 10.0  # d_min in meters
    max_uav_separation: float = 2000.0

    # Antenna configuration
    num_antennas: int = 1
    antenna_ports_per_uav: int = 1  # K=1 (fixed)

    # Optimization
    max_iterations: int = 20
    convergence_threshold: float = 0.001
    seed: int = 42

    def __post_init__(self):
        if self.stable_stage_slot is None:
            object.__setattr__(self, "stable_stage_slot", self.num_steps + 2)


@dataclass
class ObjectiveResult:
    """Result of objective function evaluation."""
    value: float  # R_S(s_T+2, k_T+2)
    iteration: int
    timestamp: float
    uav_positions: List[Position3D] = field(default_factory=list)
    per_link_rates: Dict[Tuple[int, int], float] = field(default_factory=dict)
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class OptimizationState:
    """Full state of optimization loop (LangGraph state)."""
    iteration: int = 0
    current_uav_states: List[UAVState] = field(default_factory=list)
    antenna_configs: List[AntennaConfig] = field(default_factory=list)
    current_objective: float = 0.0
    best_objective: float = 0.0
    best_uav_states: List[UAVState] = field(default_factory=list)
    proposed_positions: List[Position3D] = field(default_factory=list)
    constraints_valid: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)
    convergence_reached: bool = False
    objective_history: List[float] = field(default_factory=list)

    def copy(self) -> "OptimizationState":
        """Deep copy of state."""
        return OptimizationState(
            iteration=self.iteration,
            current_uav_states=[
                UAVState(u.uav_id, u.position, u.antenna_port)
                for u in self.current_uav_states
            ],
            antenna_configs=list(self.antenna_configs),
            current_objective=self.current_objective,
            best_objective=self.best_objective,
            best_uav_states=[
                UAVState(u.uav_id, u.position, u.antenna_port)
                for u in self.best_uav_states
            ],
            proposed_positions=list(self.proposed_positions),
            constraints_valid=self.constraints_valid,
            constraint_violations=list(self.constraint_violations),
            agent_messages=list(self.agent_messages),
            convergence_reached=self.convergence_reached,
            objective_history=list(self.objective_history),
        )
