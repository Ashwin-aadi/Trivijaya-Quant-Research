from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to exhibit less price fluctuation, which can provide "
        "more stable returns over time. Tilting the portfolio towards low-volatility stocks "
        "can help in reducing overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling standard deviation of close prices
        history = history.with_columns(
            (pl.col("adj_close").rolling_std(window=self._window)).alias("volatility")
        )
        latest_closes = view.closes(lookback=None).select(symbols)
        latest_prices = [float(latest_closes[symbol].item()) for symbol in symbols]

        # Rank symbols by volatility and select top N
        ranked_symbols = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("volatility") / pl.col("volatility").mean()).rank(method="dense", descending=True).alias("rank")
        ).sort(by="rank")

        top_n_symbols = [row["symbol"] for row in ranked_symbols.to_dicts()[:self._top_n]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest