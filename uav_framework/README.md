# Llama 3 Multi-Agent UAV Trajectory Optimization Framework

This folder contains the production-quality research prototype scaffolding.

Structure:
- simulation/     # UAV, users, environment, physics
- optimization/   # objective, optimizers (PSO, CMA-ES, etc.)
- agents/         # Planner, Evaluator, Optimizer, Environment, Termination
- llm/            # LLM wrappers and prompt templates (Groq/Ollama)
- visualization/  # Streamlit dashboard components and Plotly graphs
- config/         # Pydantic models and YAML configs
- report/         # Report generation and export
- tests/          # Unit and integration tests

Next step: implement `simulation` module after you approve this structure.