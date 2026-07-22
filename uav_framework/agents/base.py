from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
