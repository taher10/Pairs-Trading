"""
pairs_trading.py  (compatibility shim)
=======================================
The codebase has been split into three focused modules:

    data_ingestion.py  –  DataFetcher
    strategy.py        –  StrategyParams, PairsStrategy
    results.py         –  PerformanceMetrics, Backtester, ParameterOptimizer

This file re-exports everything so that any existing code that imports directly
from pairs_trading continues to work without modification.

To run the full pipeline use:
    python main.py
"""

from data_ingestion import DataFetcher
from strategy import StrategyParams, PairsStrategy
from results import PerformanceMetrics, Backtester, ParameterOptimizer

__all__ = [
    "DataFetcher",
    "StrategyParams",
    "PairsStrategy",
    "PerformanceMetrics",
    "Backtester",
    "ParameterOptimizer",
]
