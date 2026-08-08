from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression in a stock's price action indicates increased volatility and "
        "potential for breakout. High dispersion of daily prices around the mean suggests "
        "that the market is consolidating before an expected move. This can be used to identify "
        "overbought or oversold conditions, leading to potential trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_series = history[symbol]
            daily_range = (adj_close_series.high - adj_close_series.low).alias("range")
            mean_price = adj_close_series.mean().alias("mean_price")
            dispersion = (daily_range / mean_price * 100.0).alias("dispersion")

            range_comp_df = (
                history.select(
                    [symbol, "session_date", daily_range, mean_price, dispersion]
                )
                .sort("session_date", descending=False)
                .tail(self._window - 1)  # Exclude the latest date as it's not fully computed
            )

            if range_comp_df.is_empty():
                continue

            avg_dispersion = range_comp_df["dispersion"].mean().item()
            recent_dispersion = range_comp_df.tail(1)["dispersion"].item()

            if recent_dispersion >= 2 * avg_dispersion:
                picks.append(symbol)

        picks = list(set(picks))[:5]
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