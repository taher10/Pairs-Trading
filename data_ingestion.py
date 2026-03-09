"""
data_ingestion.py
=================
Responsible for downloading and aligning price data from Yahoo Finance.
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


class DataFetcher:
    """Downloads and aligns adjusted-close price data for two tickers."""

    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        start_date: str,
        end_date: str,
    ) -> None:
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns [ticker1, ticker2] indexed by date.
        Downloads both tickers in a single call so yfinance returns a clean
        MultiIndex DataFrame; we then extract the Close prices for each ticker.
        """
        raw = yf.download(
            [self.ticker1, self.ticker2],
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            group_by="ticker",
        )

        # raw[ticker]["Close"] is always a 1-D Series regardless of yfinance version
        prices1 = raw[self.ticker1]["Close"].rename(self.ticker1)
        prices2 = raw[self.ticker2]["Close"].rename(self.ticker2)

        data = pd.concat([prices1, prices2], axis=1).dropna()
        return data
