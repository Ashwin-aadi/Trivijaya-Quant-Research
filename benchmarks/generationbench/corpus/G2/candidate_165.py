from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower idiosyncratic risk and may outperform "
        "high-volatility stocks over the long term. By tilting our portfolio towards low-volatility "
        "stocks, we aim to capture this potential alpha."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            closes.sort("session_date")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .select(pl.exclude("symbol", "session_date"))
            .to_numpy()
        )

        # Calculate volatility for each symbol
        volatilities = [
            float(v.std()) for v in returns.T if not any(pl.col("r").is_nan().to_list())
        ]

        # Filter out symbols with insufficient history
        valid_symbols = [symbol for symbol, v in zip(view.symbols, volatilities) if v]

        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Rank by volatility and pick the lowest ones
        ranked_volatilities = dict(zip(valid_symbols, sorted(volatilities)))
        picks: list[str] = [k for k, v in ranked_volatilities.items() if len(ranked_volatilities) > 5]

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest