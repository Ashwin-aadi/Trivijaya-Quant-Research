from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "Combining momentum and volatility characteristics can provide a more robust signal. "
        "Momentum indicates recent price strength, while low volatility suggests stable performance."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate momentum as the percentage change from the first close to the last close
            momentum = (values[-1] - values[0]) / values[0]
            # Calculate volatility as the standard deviation of daily returns
            returns = [(v2 - v1) / v1 for v1, v2 in zip(values[:-1], values[1:])]
            volatility = pl.DataFrame({"return": returns}).with_columns(
                (pl.col("return") ** 2).alias("squared_return")
            ).select(pl.col("squared_return").mean().alias("volatility")).item()
            
            if momentum > 0.1 and volatility < 0.05:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
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