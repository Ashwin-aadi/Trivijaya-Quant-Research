from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices around "
        "key support and resistance levels relative to a trailing reference price. By "
        "identifying recent highs and lows and tracking a 50-day moving average, the strategy"
        " seeks to enter trades based on deviations from these levels."
    )

    def __init__(self, lookback: int = 14, trailing_window: int = 50, top_n: int = 20) -> None:
        self._lookback = lookback
        self._trailing_window = trailing_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=self._trailing_window + 1).filter(
            (pl.col("session_date") <= pl.lit(view.as_of))
            & (pl.col("session_date") > pl.col("session_date").dt.subtract(days=self._trailing_window - 1))
        )
        trailing_avg = history.group_by("symbol").agg(pl.col("adj_close").mean().alias("trailing_avg"))

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in trailing_avg.height_map():
                continue
            recent_highs_lows = _find_recent_extremes(closes[symbol].to_series())
            recent_high, recent_low = (
                float(recent_highs_lows["high"]),
                float(recent_highs_lows["low"]),
            )
            current_price = view.latest_close()[symbol]

            if current_price < recent_low and trailing_avg.height_map()[symbol] > recent_low:
                picks.append(symbol)
            elif current_price > recent_high and trailing_avg.height_map()[symbol] < recent_high:
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


def _find_recent_extremes(series: pl.Series) -> dict[str, float]:
    recent_highs_lows = {
        "high": series.argmax(),
        "low": series.argmin(),
    }
    high_value = series[recent_highs_lows["high"]]
    low_value = series[recent_highs_lows["low"]]
    return {"high": high_value, "low": low_value}