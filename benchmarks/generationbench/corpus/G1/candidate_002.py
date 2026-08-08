from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. High volatility periods are expected "
        "to continue trending in the same direction, allowing for profitable entries."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        vol = (
            (history["close"] / history["close"].shift(1) - 1.0).abs().mean()
        ).item()
        trend = history.sort("session_date", descending=True)["close"].head(1)[
            0
        ].item()

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            symbol_history = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(["close"])
            )

            close_values = [float(v) for v in symbol_history["close"].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue

            last_close = close_values[-1]
            mean_close = sum(close_values) / len(close_values)
            std_dev = (sum((x - mean_close) ** 2 for x in close_values) / len(close_values)) ** 0.5
            z_score = (last_close - mean_close) / std_dev

            if abs(z_score) > vol:
                picks.append(symbol)

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