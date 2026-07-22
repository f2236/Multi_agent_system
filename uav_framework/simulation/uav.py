"""UAV entity and helper utilities."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class UAV:
    uav_id: int
    pos: Tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        return np.array(self.pos, dtype=float)

    def update_from_array(self, arr: np.ndarray) -> None:
        self.pos = (float(arr[0]), float(arr[1]), float(arr[2]))
