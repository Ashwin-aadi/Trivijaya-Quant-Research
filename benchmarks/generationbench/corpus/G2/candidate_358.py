from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the NIFTY 100 may exhibit seasonality based on macroeconomic events "
        "or investor behavior. By exploiting this seasonal pattern, we can generate profitable "
        "trading signals."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
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

            # Check if the latest close is higher than the average of the same day over the past 5 years.
            daily_avg = _daily_average(view, symbol)
            if values[-1] > daily_avg:
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


def _daily_average(view: MarketView, symbol: str) -> float:
    history = view.history(lookback=252 * 5)  # 5 years of data
    daily_closes = history.select(["session_date", f"{symbol}"])
    daily_avg = (
        daily_closes.groupby("session_date").mean().sort("session_date").select(pl.col(symbol))
    ).to_numpy()[0][0]
    return float(daily_avg)