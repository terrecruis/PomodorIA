"""
edge/__init__.py — Edge AI Layer

Espone i simboli principali del layer di inferenza.
"""

from edge.inference_engine import InferenceEngine, InferenceResult

__all__ = ["InferenceEngine", "InferenceResult"]
