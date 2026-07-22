"""Streamlit app entry that wires up a simple scenario and launches the dashboard."""
import sys
from pathlib import Path
import random

try:
    import streamlit as st  # ensure Streamlit is available at runtime
except Exception:
    raise

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_framework.simulation.types import SimulationConfig
from uav_framework.simulation.environment import generate_uavs, generate_antennas
from uav_framework.agents.orchestrator import Orchestrator
from uav_framework.agents.planner import GroqPlanner
from uav_framework.agents.evaluator_agent import EvaluatorAgent
from uav_framework.optimization.objective import ObjectiveEvaluator
from uav_framework.visualization.dashboard import run_dashboard


cfg = SimulationConfig()
random.seed(0)
uavs = generate_uavs(3, cfg)
antennas = generate_antennas(1, cfg)

orch = Orchestrator(cfg, planner=GroqPlanner(cfg))
evaluator = ObjectiveEvaluator()
ev_agent = EvaluatorAgent(evaluator)
orch.set_evaluator(ev_agent)

run_dashboard(orch, uavs, antennas, iterations=8, delay=0.15)
