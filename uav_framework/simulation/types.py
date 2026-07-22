"""Pydantic models for simulation domain types."""
from pydantic import BaseModel, Field, field_validator
from typing import Tuple


class Position3D(BaseModel):
    x: float = Field(..., description="X coordinate (meters)")
    y: float = Field(..., description="Y coordinate (meters)")
    z: float = Field(..., description="Altitude (meters)")

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @field_validator("z")
    def z_must_be_finite(cls, v):
        if not (v is None or (isinstance(v, (int, float)) and v == v)):
            raise ValueError("z must be finite")
        return v


class SimulationConfig(BaseModel):
    time_step: float = Field(1.0, description="Δt in seconds")
    uav_speed: float = Field(10.0, description="UAV speed (m/s)")
    grid_x_bounds: Tuple[float, float] = Field((0.0, 1000.0))
    grid_y_bounds: Tuple[float, float] = Field((0.0, 1000.0))
    min_altitude: float = Field(5.0)
    max_altitude: float = Field(500.0)
    min_separation: float = Field(10.0)
    max_separation: float = Field(2000.0)
