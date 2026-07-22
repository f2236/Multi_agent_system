"""
Base agent class for multi-agent framework.

Abstract interface for all agents in the optimization loop.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from uav_optimizer.domain import OptimizationState


class BaseAgent(ABC):
    """Abstract base class for optimization agents."""

    def __init__(self, name: str):
        """
        Initialize agent.

        Args:
            name: Agent name identifier
        """
        self.name = name
        self.execution_count = 0

    @abstractmethod
    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute agent logic and update state.

        Args:
            state: Current optimization state

        Returns:
            Updated optimization state
        """
        pass

    def log_execution(self, state: OptimizationState, message: str) -> None:
        """
        Log agent execution message to state.

        Args:
            state: Optimization state
            message: Log message
        """
        self.execution_count += 1
        log_entry = {
            "agent": self.name,
            "execution": self.execution_count,
            "iteration": state.iteration,
            "message": message,
        }
        state.agent_messages.append(log_entry)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
