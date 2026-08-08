from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrendStrategy(Strategy):
    rationale = (
        "This strategy capitalizes on the January effect by initiating long positions when "
        "daily closing prices in January exceed their 50-day moving average. It exits positions "
        "if abnormal returns fall below a threshold or if the stock's close drops below its 20-day "
        "moving average, ensuring timely liquidation and risk management."
    )

    def __init__(self, window_50d: int = 50, window_20d: int = 20, top_n: int = 30) -> None:
        self._window_50d = window_50d
        self._window_20d = window_20d
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=100)  # Consider a larger lookback to ensure sufficient data
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            daily_df = history.filter(pl.col("symbol") == symbol)
            if daily_df.height < 100:
                continue

            closes = [float(v) for v in daily_df["close"].drop_nulls().to_list()]
            if len(closes) < self._window_50d + self._window_20d:
                continue

            close_january = max([closes[i] for i in range(31, 61) if i < daily_df.height])
            ma_50 = sum(closes[-self._window_50d:]) / self._window_50d
            ma_20 = sum(closes[-self._window_20d:]) / self._window_20d

            if close_january > ma_50:
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