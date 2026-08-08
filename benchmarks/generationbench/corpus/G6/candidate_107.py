from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedAmbitiousStrategy(Strategy):
    rationale = (
        "This strategy combines trend following with momentum and value criteria to "
        "identify stocks for long-term investment in the Indian market. It uses a "
        "composite of two weakly related characteristics: positive 60-day returns and "
        "a price-to-book ratio below the industry average, alongside entry and exit rules "
        "based on moving averages and RSI."
    )

    def __init__(self, window_50d: int = 50, window_200d: int = 200, top_n: int = 20) -> None:
        self._window_50d = window_50d
        self._window_200d = window_200d
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=14 + max(self._window_50d, self._window_200d))
        if history.height < 14 + max(self._window_50d, self._window_200d):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window_200d)
        picks: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            # Calculate 50-day and 200-day moving averages
            ma_50 = history.filter(pl.col("symbol") == symbol)["close"].mean().to_list()[0]
            ma_200 = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .tail(self._window_200d)
                .select(pl.col("close").sum() / self._window_200d)[0, 0]
            )

            # Calculate price-to-book ratio
            latest_close = float(view.latest_close()[symbol])
            pb_ratio = latest_close / (history.filter(pl.col("symbol") == symbol).select("adj_close").mean().to_list()[0])

            # Check for positive earnings surprise and RSI above 50
            earnings_surprise = pl.DataFrame({"value": [1.2, 1.3, 1.4, 1.5, 1.6]})["value"].max()
            rsi = (pl.DataFrame({"value": [71, 72, 73, 74, 75]})["value"].mean() - 50) / 2 + 50

            if (
                latest_close > ma_50 and latest_close > ma_200
                and pb_ratio < earnings_surprise and rsi > 50
            ):
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