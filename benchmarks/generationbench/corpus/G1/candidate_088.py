from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean after a strong move. By identifying symbols that have "
        "recently moved strongly and are now below their trailing average, we can generate "
        "buy signals."
    )

    def __init__(self, window: int = 30, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes().to_dict(False)

        # Calculate the trailing average
        avg_history = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("trailing_avg")))
            .collect()
        )

        symbols_with_data = set(history["symbol"]) & set(avg_history["symbol"])
        signals: dict[str, float] = {}

        for symbol in symbols_with_data:
            symbol_df = history.filter(pl.col("symbol") == symbol)
            latest_close = latest_closes[symbol][-1]
            trailing_avg = avg_history.filter(pl.col("symbol") == symbol)["trailing_avg"][0]

            if (
                latest_close < trailing_avg * (1 - self._threshold)
                and max(symbol_df["adj_close"].to_list()) > trailing_avg
            ):
                signals[symbol] = 1.0 / len(signals)

        return Signal(
            information_available_at=stamp, weights={k: v for k, v in signals.items() if v}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest