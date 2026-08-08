from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy focuses on exploiting the outperformance of low-volatility stocks in the Indian market. "
        "Historical studies suggest that stocks with lower volatility tend to provide higher risk-adjusted returns."
    )

    def __init__(self, window: int = 250, top_n_percentage: float = 0.3, bottom_n_percentage: float = 0.2) -> None:
        self._window = window
        self._top_n_percentage = top_n_percentage
        self._bottom_n_percentage = bottom_n_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and standard deviation
        returns_df = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").std().alias("volatility"))
        )

        if returns_df.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Rank by volatility
        ranked = returns_df.sort("volatility", descending=False)
        top_n = int(len(view.symbols) * self._top_n_percentage)
        bottom_n = int(len(view.symbols) * self._bottom_n_percentage)

        long_symbols = [str(row["symbol"]) for row in ranked.head(top_n).to_dicts()]
        short_symbols = [str(row["symbol"]) for row in ranked.tail(bottom_n).to_dicts()]

        # Construct weights
        if not long_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_long = 1.0 / len(long_symbols)
        signal_weights = {s: weight_long for s in long_symbols}

        if short_symbols:
            weight_short = -1.0 / len(short_symbols)
            for s in short_symbols:
                if s not in signal_weights:
                    signal_weights[s] = weight_short

        return Signal(information_available_at=stamp, weights=signal_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest