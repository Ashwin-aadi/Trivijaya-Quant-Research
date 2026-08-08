from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance during specific times of the year. "
        "By identifying these seasonal trends, we can allocate capital towards stocks that are "
        "likely to perform well in upcoming periods."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_history: dict[str, pl.DataFrame] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            symbol_df = (
                history.select(["session_date", "close"])
                .filter(pl.col("symbol") == symbol)
                .sort("session_date")
            )
            if symbol_df.height < self._window:
                continue
            symbol_history[symbol] = symbol_df

        seasonal_signals: dict[str, float] = {}
        for symbol, df in symbol_history.items():
            monthly_close = (
                df.groupby(pl.date.ExtractMonth("session_date")).agg(
                    (pl.col("close").mean()).alias("monthly_avg")
                )
            ).sort("session_date")

            if monthly_close.height < 12:
                continue

            yearly_avg = (
                monthly_close.select(["monthly_avg"])
                .mean()
                .item()
            )

            deviations = [
                (month, (row["close"] - yearly_avg) / yearly_avg)
                for month, row in monthly_close.iter_rows()
            ]

            seasonal_signals[symbol] = max(deviations, key=lambda x: abs(x[1]))[0]

        sorted_signals = [
            (symbol, score) for symbol, score in seasonal_signals.items() if score > self._threshold
        ]
        picks = [symbol for symbol, _ in sorted_signals]
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