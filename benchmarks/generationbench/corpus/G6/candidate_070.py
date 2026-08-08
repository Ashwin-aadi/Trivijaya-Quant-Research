from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class IntegratedSeasonalEquityStrategy(Strategy):
    rationale = (
        "This strategy leverages the January Effect and quarterly earnings releases to capture "
        "significant market returns. It targets specific equities during quarters with strong historical "
        "returns and uses technical analysis for entry and exit rules."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty() or len(history["symbol"].unique()) < 10:
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors = {
            "Jan-Apr": ["BANKNIFTY", "NIFTYIT"],
            "Apr-Jul": ["NIFTY50", "NIFTYPSE30"],
            "Jul-Oct": ["NIFTYPHARMA", "NIFTYFINSERV"],
            "Oct-Jan": ["NIFTYMETAL", "NIFTYGEMS"]
        }

        top_symbols = []
        for season, symbols in seasonal_factors.items():
            season_start = date(int(season[:4]), int(season[4:6]), 1)
            season_history = history.filter(
                pl.col("session_date").between(season_start, view.as_of - pl.duration(days=90))
            )
            if season_history.is_empty() or len(season_history["symbol"].unique()) < 5:
                continue

            for symbol in symbols:
                if symbol not in season_history.columns:
                    continue
                closes = season_history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol)).drop_nulls()
                if closes.height < self._window:
                    continue
                mean_close = float(closes["adj_close"].mean())
                latest_close = view.latest_close()[symbol]
                if (latest_close - mean_close) / mean_close > 0.1:
                    top_symbols.append(symbol)

        top_symbols = list(set(top_symbols))[:10]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.1
        signal_weights = {symbol: weight for symbol in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest