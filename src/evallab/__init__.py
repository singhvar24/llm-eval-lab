"""LLM Eval Lab — evaluate retrieval-augmented generation configurations."""
from .runner import Config, evaluate, load_data, pareto_front, sweep

__all__ = ["Config", "evaluate", "load_data", "pareto_front", "sweep"]
__version__ = "0.1.0"
