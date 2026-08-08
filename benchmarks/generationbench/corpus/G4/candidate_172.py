from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToMean(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their trailing price average over a specified period. "
        "When the current price diverges beyond a certain threshold, it generates buy or sell signals based on the stock's recent trend."
    )

    def __init__(self, lookback_days: int = 60, trend_days: int = 10, deviation_threshold: float = 2.0) -> None:
        self._lookback_days = lookback_days
        self._trend_days = trend_days
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days + self._trend_days)
        if history.height < self._lookback_days + self._trend_days:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_series = history.filter(pl.col("symbol") == symbol)["close"]
            trends = _simple_moving_average(close_series, self._trend_days)
            recent_trend = trends[-1]
            deviations = (close_series - trends).to_numpy()
            mean_deviation = pl.mean(deviations).item()

            if abs(mean_deviation) >= self._deviation_threshold:
                picks.append(symbol)

        picks = sorted(picks, key=lambda s: abs(_simple_moving_average(view.history(lookback=self._lookback_days).filter(pl.col("symbol") == s)["close"], 1).item() - view.latest_close()[s]), reverse=True)[:50]
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


def _simple_moving_average(series: pl.Series, window: int) -> float:
    return series.mean().item()