from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility indicates a market prone to overreactions and thus potentially "
        "predictable trend continuation. Conversely, low volatility suggests more stable "
        "markets where trends are likely to persist. This strategy leverages these phenomena by "
        "scaling trades based on realized volatility."
    )

    def __init__(self, window: int = 20, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol)
            open_vals = [float(v) for v in df["open"].to_list()]
            high_vals = [float(v) for v in df["high"].to_list()]
            low_vals = [float(v) for v in df["low"].to_list()]
            close_vals = [float(v) for v in df["close"].to_list()]

            if len(open_vals) < self._window:
                continue
            realized_volatility = (
                (pl.Series(high_vals) - pl.Series(low_vals)).abs()
                / 2.0
            ).std() * 2.0 * 252 ** 0.5

            if not picks or realized_volatility <= min(
                [realized_volatility for _, realized_volatility in picks]
            ):
                picks.append((symbol, realized_volatility))

        picks = sorted(picks, key=lambda x: x[1], reverse=True)[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_list()[0]
    assert isinstance(newest, date)
    return newest