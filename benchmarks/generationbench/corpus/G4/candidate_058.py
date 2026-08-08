from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy aims to exploit historical seasonality and calendar effects in the Indian market by "
        "identifying key dates with consistent price movements. By analyzing past returns around these dates, "
        "we aim to predict future performance and capitalize on well-documented seasonal anomalies."
    )

    def __init__(self, lookback_days: int = 30, top_n: int = 30) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_days * 2 + 1)
        if closes.height < self._lookback_days * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        key_dates = {
            date(2023, 11, 14): "Diwali",  # Example date for Diwali
            date(2023, 3, 31): "March-end",
            date(2023, 5, 17): "Weekday Effect"  # Example date for a weekday effect
        }

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            returns: list[float] = []

            for key_date, event_name in key_dates.items():
                start_date = max(key_date - self._lookback_days * 2, stamp)
                end_date = min(key_date + self._lookback_days, view.as_of)

                if not (start_date <= view.as_of < end_date):
                    continue

                df_event = closes.filter(
                    pl.col("session_date") >= start_date
                    & pl.col("session_date") <= end_date
                    & pl.col(symbol).is_not_null()
                )

                if df_event.is_empty():
                    continue

                returns.append(df_event.sort("session_date").select(
                    (pl.col("adj_close").shift(-1) / pl.col("adj_close").first() - 1.0)
                ).collect().height > 0)

            avg_return = sum(returns) / len(key_dates)
            if avg_return > 0:
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