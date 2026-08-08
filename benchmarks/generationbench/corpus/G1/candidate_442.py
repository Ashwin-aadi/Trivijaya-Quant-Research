from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets often reflects recurring events or behaviors that affect "
        "stock prices. By identifying stocks with historical trends during specific times of the year, "
        "investors can capitalize on these patterns."
    )

    def __init__(self, window: int = 365, seasonality_window: int = 90) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        seasonality_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._seasonality_window:
                continue

            # Calculate the average close price for each month within the seasonality window
            monthly_closes = (
                history.select(
                    pl.col("session_date").dt.month_name(),
                    pl.col(symbol),
                )
                .group_by(pl.col("session_date").dt.month_name())
                .agg([pl.col(symbol).mean()])
                .collect()
            )

            # Compute the average close price for each month over the past year
            mean_monthly_closes = (
                monthly_closes.sort("session_date")
                .select(
                    pl.col("month").shift_and_fill(self._seasonality_window - 1),
                    pl.col(symbol).mean().alias("average"),
                )
                .sort("month", descending=False)
                .collect()
            )

            # Calculate the seasonality score as the difference between current close and mean of past year
            current_close = values[-1]
            if not mean_monthly_closes.is_empty():
                latest_mean_close = float(mean_monthly_closes.select("average").to_list()[-1])
                seasonality_scores[symbol] = abs(current_close - latest_mean_close)

        top_symbols = [s for s, score in sorted(seasonality_scores.items(), key=lambda x: x[1], reverse=True)][:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest