"""
Configuration management for UAV trajectory optimization.

Loads defaults and can be overridden via YAML config files.
"""

import os
from typing import Dict, Any, Optional, List
import yaml
from dataclasses import asdict
from uav_optimizer.domain import SimulationConfig, AntennaConfig, Position3D


class Config:
    """Singleton configuration manager."""

    _instance: Optional["Config"] = None
    _simulation_config: SimulationConfig
    _antennas: List[AntennaConfig]
    _debug: bool

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize with defaults."""
        if not hasattr(self, "_initialized"):
            self._simulation_config = self._default_simulation_config()
            self._antennas = self._default_antennas()
            self._debug = os.getenv("DEBUG", "false").lower() == "true"
            self._initialized = True

    @staticmethod
    def _default_simulation_config() -> SimulationConfig:
        """Create default simulation config."""
        return SimulationConfig(
            time_step_duration=1.0,
            num_steps=10,
            uav_speed=10.0,
            min_altitude=5.0,
            max_altitude=500.0,
            grid_x_bounds=(0.0, 1000.0),
            grid_y_bounds=(0.0, 1000.0),
            min_uav_separation=10.0,
            max_uav_separation=2000.0,
            num_antennas=1,
            antenna_ports_per_uav=1,
            max_iterations=20,
            convergence_threshold=0.001,
            seed=42,
        )

    @staticmethod
    def _default_antennas() -> List[AntennaConfig]:
        """Create default antenna configuration."""
        return [
            AntennaConfig(
                antenna_id=0,
                position=Position3D(x=500.0, y=500.0, z=20.0),
                max_users=10,
                tx_power_dbm=20.0,
                bandwidth_hz=1e6,
                frequency_hz=2.4e9,
                noise_figure_db=7.0,
            )
        ]

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            return

        # Load simulation config
        if "simulation" in data:
            sim_data = data["simulation"]
            self._simulation_config = SimulationConfig(
                time_step_duration=sim_data.get("time_step_duration", 1.0),
                num_steps=sim_data.get("num_steps", 10),
                uav_speed=sim_data.get("uav_speed", 10.0),
                min_altitude=sim_data.get("min_altitude", 5.0),
                max_altitude=sim_data.get("max_altitude", 500.0),
                grid_x_bounds=tuple(sim_data.get("grid_x_bounds", [0.0, 1000.0])),
                grid_y_bounds=tuple(sim_data.get("grid_y_bounds", [0.0, 1000.0])),
                min_uav_separation=sim_data.get("min_uav_separation", 10.0),
                max_uav_separation=sim_data.get("max_uav_separation", 2000.0),
                num_antennas=sim_data.get("num_antennas", 1),
                antenna_ports_per_uav=sim_data.get("antenna_ports_per_uav", 1),
                max_iterations=sim_data.get("max_iterations", 20),
                convergence_threshold=sim_data.get("convergence_threshold", 0.001),
                seed=sim_data.get("seed", 42),
            )

        # Load antennas
        if "antennas" in data:
            self._antennas = []
            for ant_data in data["antennas"]:
                ant = AntennaConfig(
                    antenna_id=ant_data["antenna_id"],
                    position=Position3D(
                        x=ant_data["position"]["x"],
                        y=ant_data["position"]["y"],
                        z=ant_data["position"]["z"],
                    ),
                    max_users=ant_data.get("max_users", 10),
                    tx_power_dbm=ant_data.get("tx_power_dbm", 20.0),
                    bandwidth_hz=ant_data.get("bandwidth_hz", 1e6),
                    frequency_hz=ant_data.get("frequency_hz", 2.4e9),
                    noise_figure_db=ant_data.get("noise_figure_db", 7.0),
                )
                self._antennas.append(ant)

        if "debug" in data:
            self._debug = data["debug"]

    @property
    def simulation(self) -> SimulationConfig:
        """Get simulation configuration."""
        return self._simulation_config

    @property
    def antennas(self) -> List[AntennaConfig]:
        """Get antenna configurations."""
        return self._antennas

    @property
    def debug(self) -> bool:
        """Is debug mode enabled."""
        return self._debug

    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "simulation": asdict(self._simulation_config),
            "antennas": [asdict(a) for a in self._antennas],
            "debug": self._debug,
        }
