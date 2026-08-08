from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining the strength of 20-day momentum with the volatility of 10-day price "
        "range provides a more robust entry signal by leveraging both trend and market "
        "volatility."
    )

    def __init__(self, momentum_window: int = 20, vol_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._vol_window)
        if history.height < self._momentum_window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_ranks: list[str] = []
        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = history[symbol]
            if len(close_series.to_list()) < self._momentum_window + self._vol_window:
                continue

            momentum_close = float(close_series[-1])
            momentum_rank = _ranked_strength(momentum_close, close_series)

            daily_range = (
                pl.col("high") - pl.col("low")
            ).mean().item()
            volatility = (daily_range / close_series.iloc[0]).log2()

            if momentum_rank >= self._top_n_momentum and volatilities.get(symbol) < volatility:
                momentum_ranks.append(symbol)
                volatilities[symbol] = volatility

        picks = [symbol for symbol in momentum_ranks if volatilities[symbol] > 1.0]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _ranked_strength(close: float, series: pl.Series) -> int:
    rank = (
        (series.rank(method="dense", descending=True)["adj_close"] == 1).filter(
            pl.col("adj_close") <= close
        )
        .height
    )
    return max(rank, 1)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest