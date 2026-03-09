"""
main.py
=======
Entry point for the pairs trading strategy.
Wires together data ingestion, strategy execution, and results reporting.

Outputs (written to output/)
-----------------------------
  default_backtest_chart.png  – Spread + PnL chart for the default parameters
  best_backtest_chart.png     – Spread + PnL chart for the best found parameters
  results.xlsx                – Three sheets:
                                  Default Backtest | Grid Search | Best Backtest

Run
---
    python main.py
"""

from pathlib import Path

import pandas as pd

from data_ingestion import DataFetcher
from results import Backtester, ParameterOptimizer
from strategy import StrategyParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKER1      = "AAPL"
TICKER2      = "MSFT"
START_DATE   = "2020-01-01"
END_DATE     = "2025-01-31"
TRAIN_LENGTH = 1008
OUTPUT_DIR   = Path("output")


def _metrics_to_df(label: str, params: StrategyParams, metrics: dict) -> pd.DataFrame:
    """Convert a single backtest result into a one-row DataFrame."""
    return pd.DataFrame([{
        "label":              label,
        "long_threshold":     params.long_threshold,
        "short_threshold":    params.short_threshold,
        "exit_threshold":     params.exit_threshold,
        "hedge_ratio_lookback": params.hedge_ratio_lookback,
        "lookback_window":    params.lookback_window,
        "sharpe":             round(metrics["sharpe"], 4),
        "max_drawdown":       round(metrics["max_drawdown"], 4),
        "drawdown_duration":  metrics["drawdown_duration"],
        "total_return":       round(metrics["total_return"], 4),
    }])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    print(f"Fetching data for {TICKER1} and {TICKER2}...")
    data = DataFetcher(TICKER1, TICKER2, START_DATE, END_DATE).fetch()
    print(
        f"  {len(data)} trading days loaded "
        f"({data.index[0].date()} \u2192 {data.index[-1].date()})\n"
    )

    backtester = Backtester(
        data, TICKER1, TICKER2,
        train_length=TRAIN_LENGTH,
        output_dir=OUTPUT_DIR,
    )

    # ── 2. Default backtest ────────────────────────────────────────────────
    default_params  = StrategyParams()
    default_metrics = backtester.run(default_params, label="Default Backtest")
    default_df      = _metrics_to_df("Default", default_params, default_metrics)

    # ── 3. Grid search ─────────────────────────────────────────────────────
    optimizer  = ParameterOptimizer(backtester)
    best_params = optimizer.optimise()
    grid_df    = optimizer.results_dataframe()

    # ── 4. Best-params backtest ────────────────────────────────────────────
    best_metrics = backtester.run(best_params, label="Best Backtest")
    best_df      = _metrics_to_df("Best", best_params, best_metrics)

    # ── 5. Export to Excel ─────────────────────────────────────────────────
    excel_path = OUTPUT_DIR / "results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        default_df.to_excel(writer, sheet_name="Default Backtest", index=False)
        grid_df.to_excel(writer, sheet_name="Grid Search",         index=False)
        best_df.to_excel(writer, sheet_name="Best Backtest",       index=False)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print(f"  {OUTPUT_DIR}/Default_Backtest_chart.png")
    print(f"  {OUTPUT_DIR}/Best_Backtest_chart.png")
    print(f"  {excel_path}")


if __name__ == "__main__":
    main()
