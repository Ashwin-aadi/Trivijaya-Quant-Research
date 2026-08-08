from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BollingerBandsReversion(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices "
        "around certain key price levels. By identifying critical support and resistance "
        "levels through historical data analysis, the strategy enters trades betting that "
        "prices will return to their mean level after breaching these levels."
    )

    def __init__(self, window: int = 200, factor: float = 2.0) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate mean and standard deviation
        mean_adj_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).to_series()[0]
        std_adj_close = history.select(
            (pl.col("adj_close") - mean_adj_close).stddev().alias("std")
        ).to_series()[0]

        # Calculate Bollinger Bands
        upper_band = mean_adj_close + self._factor * std_adj_close
        lower_band = mean_adj_close - self._factor * std_adj_close

        # Identify key support and resistance levels
        support_levels = history.with_columns(
            (pl.col("adj_close") < lower_band).alias("below_lower")
        )
        resistance_levels = history.with_columns(
            (pl.col("adj_close") > upper_band).alias("above_upper")
        )

        # Find breaches in the last session date
        latest_session_history = support_levels.filter(
            pl.col("session_date") == view.as_of
        ).to_series()["below_lower"]
        latest_session_resistance = resistance_levels.filter(
            pl.col("session_date") == view.as_of
        ).to_series()["above_upper"]

        breaches: list[str] = []
        for symbol in view.symbols:
            if latest_session_history[symbol]:
                breaches.append(symbol)
            elif latest_session_resistance[symbol]:
                breaches.append(symbol)

        # Rank and select top N symbols based on deviation from the mean
        ranked_breaches = (
            history.select(
                pl.col("symbol"),
                ((pl.col("adj_close") - mean_adj_close).abs().alias("deviation")),
            )
            .filter(pl.col("session_date") == view.as_of)
            .sort("deviation", descending=True)
            .limit(20)
        )

        selected_symbols = [row["symbol"] for row in ranked_breaches]
        if not breaches:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breaches)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series()[0]
    assert isinstance(newest, date)
    return newest