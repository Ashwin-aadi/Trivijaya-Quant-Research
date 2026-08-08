from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Seasonality in equity markets can arise from predictable changes in economic "
        "conditions or investor behavior throughout the year. For instance, certain sectors "
        "may experience increased activity during specific months due to holidays or weather patterns."
    )

    def __init__(self, window: int = 30, seasonality_periods: list[int] = [12]) -> None:
        self._window = window
        self._seasonality_periods = seasonality_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        for period in self._seasonality_periods:
            seasonal_effects = (
                (history.with_columns(
                    ((pl.col("session_date") - pl.col("session_date").dt.year().first())
                     / 30.0) % period).group_by(["symbol"]).agg([
                        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)
                        .alias(f"return_{period}")
                    ])
                ).sort("session_date").tail(period)
            )

            if seasonal_effects.height > 0:
                top_performer = seasonal_effects.sort(
                    f"return_{period}", descending=True
                )[f"return_{period}"].to_list()[0]
                break

        else:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_close = float(view.latest_close().get(symbol) or 0.0)
            if (recent_close / recent_close * top_performer - 1.0) > 0.02:
                picks.append(symbol)

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