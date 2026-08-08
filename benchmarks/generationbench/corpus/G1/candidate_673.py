from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that asset prices will revert to the average "
        "price level after a deviation from it. We can exploit this by identifying stocks that"
        " have moved significantly away from their recent average price and betting on them "
        "to return to the mean."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation")
            )
        )

        signal_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in means.columns:
                continue
            mean_deviation = float(means[means["symbol"] == symbol]["deviation"])
            if mean_deviation >= self._threshold * (float(means[means["symbol"] == symbol]["mean"]) / 100):
                signal_symbols.append(symbol)

        weights = {s: 1.0 / len(signal_symbols) for s in signal_symbols}
        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest