from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Dispersion5d(Strategy):
    rationale = (
        "Range compression (difference between high and low) is a proxy for short-term volatility. "
        "An increase in dispersion indicates heightened volatility and potential trading opportunities."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        dispersions: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].select(
                (pl.col("high") - pl.col("low")).alias("dispersion")
            ).select("dispersion").to_list()]
            if len(values) < self._window:
                continue

            latest_dispersion = values[-1]
            prev_mean_displacement = sum(values[:-1]) / len(values[:-1])
            if (latest_dispersion - prev_mean_displacement) > 0.0:
                dispersions[symbol] = latest_dispersion

        picks = sorted(dispersions.keys(), key=lambda k: dispersions[k], reverse=True)[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest