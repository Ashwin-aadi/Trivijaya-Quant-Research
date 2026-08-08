from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 50-day moving "
        "average of the stock price and the number of consecutive sessions where the volume "
        "exceeds its 21-day average. These indicators are used to identify stocks with both"
        " strong momentum and sustained interest."
    )

    def __init__(self, ma_window: int = 50, vol_window: int = 21) -> None:
        self._ma_window = ma_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._ma_window, self._vol_window))
        if history.is_empty() or history.height < max(self._ma_window, self._vol_window):
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.symbol.unique()]
        signals: list[str] = []

        for symbol in symbols:
            ma_close = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    (pl.col("adj_close").mean().over("session_date").alias(f"ma_{self._ma_window}"))
                )
                .sort("session_date", descending=True)
                .with_column(
                    (pl.col(f"ma_{self._ma_window}") / pl.col("adj_close") - 1.0).alias("ma_diff")
                )["ma_diff"].to_list()
            )

            if len(ma_close) < self._ma_window:
                continue

            ma_condition = all([abs(val) <= 0.1 for val in ma_close[-self._ma_window:]])

            vol_history = (
                history.filter(pl.col("symbol") == symbol)
                .group_by("session_date")
                .agg(
                    (pl.col("volume").mean().alias(f"vol_{self._vol_window}"))
                )
            )

            if len(vol_history) < self._vol_window:
                continue

            vol_diffs = (
                vol_history.with_column(
                    (pl.col("volume") / pl.col(f"vol_{self._vol_window}") - 1.0).alias("vol_ratio")
                )["vol_ratio"].to_list()
            )

            vol_condition = any([abs(val) > 0.2 for val in vol_diffs[-self._vol_window:]])

            if ma_condition and vol_condition:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest