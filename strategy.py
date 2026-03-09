"""
strategy.py
===========
Core pairs trading logic: hyperparameters, hedge ratio estimation,
spread/z-score computation, signal generation, and PnL calculation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    """All tunable hyperparameters for the pairs trading strategy."""

    long_threshold: float = -2.0       # Enter long spread below this z-score
    short_threshold: float = 2.0       # Enter short spread above this z-score
    exit_threshold: float = 0.5        # Exit when |z-score| falls below this
    hedge_ratio_lookback: int = 100    # Rolling window length for OLS hedge ratio
    lookback_window: int = 100         # Rolling window length for spread mean/std

    def __str__(self) -> str:
        return (
            f"long={self.long_threshold}, short={self.short_threshold}, "
            f"exit={self.exit_threshold}, hedge_lookback={self.hedge_ratio_lookback}, "
            f"spread_lookback={self.lookback_window}"
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class PairsStrategy:
    """
    Implements the pairs trading pipeline:
      1. Rolling OLS to estimate the hedge ratio.
      2. Spread = asset1 - hedge_ratio * asset2.
      3. Rolling z-score of the spread.
      4. Signal generation and position construction (with ffill).
      5. Daily PnL with a one-day position lag to avoid look-ahead bias.

    Usage
    -----
        strategy = PairsStrategy(params).fit(prices1, prices2)
        pnl = strategy.pnl
    """

    def __init__(self, params: StrategyParams) -> None:
        self.params = params

        # Populated after calling fit()
        self.hedge_ratios: Optional[np.ndarray] = None
        self.spread: Optional[np.ndarray] = None
        self.zscore: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.daily_returns: Optional[np.ndarray] = None
        self.pnl: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, prices1: np.ndarray, prices2: np.ndarray) -> "PairsStrategy":
        """Run the full strategy pipeline on raw price arrays."""
        self.hedge_ratios = self._rolling_hedge_ratio(prices1, prices2)
        self.spread = self._compute_spread(prices1, prices2)
        self.zscore = self._compute_zscore(self.spread)
        self.positions = self._generate_positions(self.zscore)
        self.daily_returns = self._compute_daily_returns(prices1, prices2)
        self.pnl = self._compute_pnl(self.positions, self.daily_returns)
        return self

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _rolling_hedge_ratio(
        self, prices1: np.ndarray, prices2: np.ndarray
    ) -> np.ndarray:
        """Estimate hedge ratio via rolling OLS: prices1 ~ beta * prices2 (no intercept)."""
        lookback = self.params.hedge_ratio_lookback
        n = len(prices1)
        hedge_ratios = np.full(n, np.nan)

        for i in range(lookback, n):
            y = prices1[i - lookback : i]
            X = prices2[i - lookback : i].reshape(-1, 1)
            hedge_ratios[i] = sm.OLS(y, X, hasconst=False).fit().params[0]

        # Back-fill initial window with the first valid estimate
        hedge_ratios[:lookback] = hedge_ratios[lookback]
        return hedge_ratios

    def _compute_spread(
        self, prices1: np.ndarray, prices2: np.ndarray
    ) -> np.ndarray:
        """Spread = prices1 - hedge_ratio * prices2."""
        return prices1 - self.hedge_ratios * prices2

    def _compute_zscore(self, spread: np.ndarray) -> np.ndarray:
        """Normalise the spread with a rolling mean and standard deviation."""
        s = pd.Series(spread)
        window = self.params.lookback_window
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
        return ((s - mean) / std).values

    def _generate_positions(self, zscore: np.ndarray) -> np.ndarray:
        """
        Entry / exit rules based on z-score thresholds:
          z <= long_threshold   → long spread  (+1 asset1, -1 asset2)
          z >= short_threshold  → short spread (-1 asset1, +1 asset2)
          |z| <= exit_threshold → flat (0, 0)

        Rows with no new signal are NaN so they forward-fill from the prior
        position. Exit rows are explicitly set to 0 AFTER forward-filling so
        they correctly close any open position (not overwritten by ffill).
        """
        p = self.params
        n = len(zscore)

        # Start with NaN = "no new decision, hold prior position"
        pos1 = np.full(n, np.nan)
        pos2 = np.full(n, np.nan)

        long_mask  = zscore <= p.long_threshold
        short_mask = zscore >= p.short_threshold
        exit_mask  = np.abs(zscore) <= p.exit_threshold

        pos1[long_mask]  =  1;  pos2[long_mask]  = -1
        pos1[short_mask] = -1;  pos2[short_mask] =  1
        # Exit signals override any entry on the same bar
        pos1[exit_mask]  =  0;  pos2[exit_mask]  =  0

        # ffill carries the active position forward until an exit or reversal
        positions = (
            pd.DataFrame({"p1": pos1, "p2": pos2})
            .ffill()
            .fillna(0)   # flat before the first signal
            .values
        )
        return positions

    def _compute_daily_returns(
        self, prices1: np.ndarray, prices2: np.ndarray
    ) -> np.ndarray:
        """Daily percentage returns for both assets stacked column-wise."""
        prices = np.column_stack([prices1, prices2])
        prices_prev = np.roll(prices, shift=1, axis=0).astype(float)
        prices_prev[0, :] = np.nan
        return (prices - prices_prev) / prices_prev

    def _compute_pnl(
        self, positions: np.ndarray, daily_returns: np.ndarray
    ) -> np.ndarray:
        """Daily PnL = lagged position * current-day return (avoids look-ahead bias)."""
        positions_prev = np.roll(positions, shift=1, axis=0)
        positions_prev[0, :] = 0
        return np.nansum(positions_prev * daily_returns, axis=1)
