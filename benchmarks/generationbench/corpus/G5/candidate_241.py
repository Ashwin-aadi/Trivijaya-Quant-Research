from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies seek to capitalize on deviations from an asset's historical "
        "mean price. When prices deviate significantly from the mean in a short time frame, "
        "there is often a tendency for them to revert to their average value."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean()).collect().get(0).item(float)
        )
        reversion_symbols = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            recent_close = values[-1]
            z_score = (recent_close - mean_close) / (
                history.select(pl.col("adj_close").stddev()).collect().get(0).item(float)
                or 1e-6  # Avoid division by zero
            )
            if abs(z_score) > 2.0:
                reversion_symbols.append(symbol)

        weight = 1.0 / len(reversion_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in reversion_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest