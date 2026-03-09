"""
results.py
==========
Performance measurement, backtesting, visualisation, and parameter optimisation.

Classes
-------
PerformanceMetrics – Static risk/return metrics (Sharpe, drawdown, IR).
Backtester         – Runs train/test split, saves charts to output/, exports Excel.
ParameterOptimizer – Grid-searches StrategyParams to maximise test-set Sharpe.
"""

from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive – no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy import PairsStrategy, StrategyParams

warnings.filterwarnings("ignore")

DEFAULT_OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

class PerformanceMetrics:
    """Collection of static risk/return metrics."""

    @staticmethod
    def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio (zero risk-free rate assumed)."""
        mu = np.nanmean(returns)
        sigma = np.nanstd(returns)
        if sigma == 0:
            return np.nan
        return float(np.sqrt(periods_per_year) * mu / sigma)

    @staticmethod
    def max_drawdown(pnl: np.ndarray) -> tuple[float, int]:
        """
        Return (max_drawdown, drawdown_duration_in_days).
        Max drawdown is expressed as a negative value.
        """
        cumulative = np.nancumsum(pnl)
        running_peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_peak
        max_dd = float(np.min(drawdown))

        if max_dd < 0:
            trough_idx = int(np.where(drawdown == max_dd)[0][-1])
            peak_val = running_peak[trough_idx]
            peak_indices = np.where(cumulative == peak_val)[0]
            peak_idx = int(peak_indices[0]) if len(peak_indices) else 0
            duration = max(trough_idx - peak_idx, 1)
        else:
            duration = 0

        return max_dd, duration

    @staticmethod
    def information_ratio(pnl: np.ndarray, benchmark_pnl: np.ndarray) -> float:
        """Information ratio: mean excess return divided by tracking error."""
        excess = pnl - benchmark_pnl
        std_excess = np.nanstd(excess)
        if std_excess == 0:
            return np.nan
        return float(np.nanmean(excess) / std_excess)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Splits data into train and test sets, runs the strategy, saves charts to
    output/<label>_chart.png, and collects metrics for Excel export.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        train_length: int = 1008,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.data = data
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.train_length = train_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        n = len(data)
        self.trainset = np.arange(0, train_length)
        self.testset = np.arange(train_length, n)

    def run(self, params: StrategyParams, label: str = "backtest") -> dict:
        """
        Execute the strategy, print a metrics summary, save a chart to
        output/<label>_chart.png, and return a metrics dictionary.
        """
        prices1 = self.data[self.ticker1].values.ravel().astype(float)
        prices2 = self.data[self.ticker2].values.ravel().astype(float)
        dates   = self.data.index

        strategy = PairsStrategy(params).fit(prices1, prices2)
        pnl_test = strategy.pnl[self.testset]

        metrics = self._compute_metrics(pnl_test)
        self._print_summary(label, params, metrics)
        chart_path = self._save_chart(strategy.spread, pnl_test, dates, label)
        print(f"  Chart saved  : {chart_path}")

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, pnl_test: np.ndarray) -> dict:
        sharpe = PerformanceMetrics.annualized_sharpe(pnl_test)
        max_dd, dd_duration = PerformanceMetrics.max_drawdown(pnl_test)
        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "drawdown_duration": dd_duration,
            "total_return": float(np.nansum(pnl_test)),
        }

    def _print_summary(self, label: str, params: StrategyParams, metrics: dict) -> None:
        print("=" * 55)
        print(f"  {label}")
        print("=" * 55)
        print(f"  Parameters   : {params}")
        print(f"  Sharpe Ratio : {metrics['sharpe']:.4f}")
        print(f"  Max Drawdown : {metrics['max_drawdown']:.4f}")
        print(f"  DD Duration  : {metrics['drawdown_duration']} days")
        print(f"  Total Return : {metrics['total_return']:.4f}")
        print("=" * 55)

    def _save_chart(
        self, spread: np.ndarray, pnl_test: np.ndarray, dates: pd.DatetimeIndex, label: str
    ) -> Path:
        """Save a two-panel spread + cumulative PnL chart as a PNG."""
        train_dates = dates[self.trainset]
        test_dates  = dates[self.testset]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Spread – train vs test, both on a real date axis
        axes[0].plot(train_dates, spread[self.trainset], label="Train", color="steelblue")
        axes[0].plot(test_dates,  spread[self.testset],  label="Test",  color="darkorange")
        axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
        axes[0].set_title(f"Spread Over Time  [{label}]")
        axes[0].set_ylabel("Spread (residual)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        fig.autofmt_xdate(rotation=30)

        # Cumulative PnL on test set with real dates
        axes[1].plot(test_dates, np.nancumsum(pnl_test), label="Cumulative PnL (Test)", color="seagreen")
        axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
        axes[1].set_title(f"Test Set Cumulative PnL  [{label}]")
        axes[1].set_ylabel("Cumulative PnL")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.autofmt_xdate(rotation=30)

        plt.tight_layout()
        path = self.output_dir / f"{label.replace(' ', '_')}_chart.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# Parameter optimiser
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """
    Grid-searches over a parameter space to find the StrategyParams that
    maximise the test-set Sharpe ratio.
    """

    DEFAULT_GRID: dict = {
        "long_threshold": [-2, -1],
        "short_threshold": [1, 2],
        "exit_threshold": [0.5, 1.0],
        "hedge_ratio_lookback": [100, 200, 252],
        "lookback_window": [100, 200, 252],
    }

    def __init__(
        self,
        backtester: Backtester,
        param_grid: Optional[dict] = None,
    ) -> None:
        self.backtester = backtester
        self.param_grid = param_grid or self.DEFAULT_GRID

        self.best_sharpe: float = -np.inf
        self.best_params: Optional[StrategyParams] = None
        self.results: list[dict] = []

    def optimise(self) -> StrategyParams:
        """
        Iterate over all parameter combinations and return the best StrategyParams.
        Progress and per-combination Sharpe ratios are printed to stdout.
        Full grid results are stored in self.results for Excel export.
        """
        keys = list(self.param_grid.keys())
        combinations = list(product(*self.param_grid.values()))
        total = len(combinations)
        print(f"Running grid search over {total} parameter combinations...\n")

        prices1 = self.backtester.data[self.backtester.ticker1].values.ravel().astype(float)
        prices2 = self.backtester.data[self.backtester.ticker2].values.ravel().astype(float)

        for i, values in enumerate(combinations, start=1):
            params = StrategyParams(**dict(zip(keys, values)))

            strategy = PairsStrategy(params).fit(prices1, prices2)
            pnl_test = strategy.pnl[self.backtester.testset]
            sharpe = PerformanceMetrics.annualized_sharpe(pnl_test)
            max_dd, dd_dur = PerformanceMetrics.max_drawdown(pnl_test)

            self.results.append({
                "rank": i,
                "long_threshold": params.long_threshold,
                "short_threshold": params.short_threshold,
                "exit_threshold": params.exit_threshold,
                "hedge_ratio_lookback": params.hedge_ratio_lookback,
                "lookback_window": params.lookback_window,
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
                "drawdown_duration": dd_dur,
                "total_return": round(float(np.nansum(pnl_test)), 4),
            })

            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = params

            print(f"  [{i:>4}/{total}] Sharpe={sharpe:.4f}  |  {params}")

        print(f"\nBest Sharpe : {self.best_sharpe:.4f}")
        print(f"Best Params : {self.best_params}")
        return self.best_params

    def results_dataframe(self) -> pd.DataFrame:
        """Return the grid search results as a sorted DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return (
            pd.DataFrame(self.results)
            .sort_values("sharpe", ascending=False)
            .reset_index(drop=True)
        )
