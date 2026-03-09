# Pairs Trading Strategy

A mean-reversion statistical arbitrage strategy that trades the spread between two correlated equities. The spread is estimated using a rolling OLS hedge ratio (with intercept) and normalised into a z-score to generate long/short signals.

---

## Strategy Overview

### What is Pairs Trading?

Pairs trading exploits the tendency of two historically correlated assets to move together. When their prices temporarily diverge, the strategy bets that the gap will close — buying the underperformer and shorting the outperformer, then exiting when the spread reverts to its mean.

### How It Works — Step by Step

**1. Price Data Ingestion**  
Adjusted close prices for two tickers are downloaded from Yahoo Finance over a defined date range and aligned to shared trading dates.

**2. Rolling Hedge Ratio (OLS with Intercept)**  
At each time step, the hedge ratio β is estimated over a rolling window using Ordinary Least Squares:

```
Asset1 = α + β × Asset2 + ε
```

Including the intercept (α) is critical — it prevents β from absorbing the mean-level price difference between the two assets, which would otherwise make the spread non-stationary. The intercept is also used when computing the spread.

**3. Spread Computation**  
The spread is the OLS residual:

```
Spread = Asset1 − (α + β × Asset2)
```

A well-estimated spread oscillates around zero, making it suitable for mean-reversion trading.

**4. Rolling Z-Score**  
The spread is normalised into a z-score using a rolling mean and standard deviation:

```
Z = (Spread − RollingMean) / RollingStd
```

This makes the signal scale-invariant and interpretable in terms of standard deviations from the mean.

**5. Signal Generation**  

| Condition | Action |
|---|---|
| Z ≤ `long_threshold` | **Long spread** — buy Asset1, short Asset2 |
| Z ≥ `short_threshold` | **Short spread** — short Asset1, buy Asset2 |
| \|Z\| ≤ `exit_threshold` | **Flat** — close all positions |

Positions are forward-filled between signals, meaning an active trade is held until an explicit exit signal fires.

**6. PnL Calculation**  
Daily PnL is computed using yesterday's position and today's return to avoid look-ahead bias:

```
PnL(t) = Position(t−1) × DailyReturn(t)
```

---

## Backtesting Methodology

The data is split into two non-overlapping periods:

| Period | Purpose |
|---|---|
| **Training Set** | First 1008 trading days (~4 years). Used to fit the hedge ratio and z-score. No trades evaluated here. |
| **Test Set** | Remaining days (~1 year). All performance metrics are evaluated exclusively on this period. |

This prevents in-sample overfitting — the model never sees test data during fitting.

### Baseline Strategy

The first backtest uses a fixed set of hand-chosen parameters:

| Parameter | Default Value |
|---|---|
| `long_threshold` | −2.0 |
| `short_threshold` | 2.0 |
| `exit_threshold` | 0.5 |
| `hedge_ratio_lookback` | 100 days |
| `lookback_window` | 100 days |

This establishes a performance benchmark before any tuning.

### Optimised Strategy

A full grid search is run over all combinations in the parameter grid. Each combination is evaluated on the **test set Sharpe ratio** (not training), and the best-performing set of parameters is used to produce the final strategy run.

This is the strategy's best achievable configuration within the defined search space.

---

## Performance Metrics

| Metric | Description |
|---|---|
| **Sharpe Ratio** | Annualised return-to-risk ratio (√252 × mean daily PnL / std daily PnL) |
| **Max Drawdown** | Largest peak-to-trough decline in cumulative PnL |
| **Drawdown Duration** | Number of days from peak to trough of the worst drawdown |
| **Total Return** | Sum of daily PnL over the test period |

---

## Project Structure

```
Pairs-Trading/
├── data_ingestion.py   # DataFetcher — downloads & aligns price data
├── strategy.py         # StrategyParams, PairsStrategy — hedge ratio, spread, signals, PnL
├── results.py          # PerformanceMetrics, Backtester, ParameterOptimizer
├── main.py             # Entry point — wires all modules together
├── pairs_trading.py    # Compatibility shim (re-exports all classes)
├── requirements.txt    # Python dependencies
├── output/             # Generated on run (charts + Excel report)
│   ├── Baseline_Strategy_chart.png
│   ├── Optimised_Strategy_chart.png
│   └── results.xlsx    # Sheets: Baseline Strategy | Grid Search | Optimised Strategy
└── .venv/              # Virtual environment (not committed)
```

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/taher10/Pairs-Trading.git
cd Pairs-Trading
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the strategy**
```bash
python3 main.py
```

This will:
1. Download price data for AAPL and MSFT (2020–2025)
2. Run the **Baseline Strategy** with default parameters
3. Run a **Grid Search** over 72 parameter combinations
4. Run the **Optimised Strategy** with the best found parameters
5. Save two PNG charts and a results Excel file to `output/`

---

## Configuration

Edit the constants at the top of [main.py](main.py) to change tickers, date range, or train length:

```python
TICKER1      = "AAPL"
TICKER2      = "MSFT"
START_DATE   = "2020-01-01"
END_DATE     = "2025-01-31"
TRAIN_LENGTH = 1008        # trading days in the training set
```

To customise the grid search, pass a `param_grid` dict to `ParameterOptimizer`:

```python
from results import ParameterOptimizer

optimizer = ParameterOptimizer(backtester, param_grid={
    "long_threshold":       [-3, -2, -1],
    "short_threshold":      [1, 2, 3],
    "exit_threshold":       [0.25, 0.5, 1.0],
    "hedge_ratio_lookback": [60, 100, 252],
    "lookback_window":      [60, 100, 252],
})
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Market data download |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | Rolling OLS regression |
| `matplotlib` | Chart generation |
| `openpyxl` | Excel export |
