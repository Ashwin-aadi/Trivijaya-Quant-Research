from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the Indian market show historically higher returns during specific seasons "
        "of the year. This strategy aims to capture these seasonal effects by allocating capital towards "
        "stocks that have performed well in previous years during similar periods."
    )

    def __init__(self, window: int = 365, season_length: int = 90) -> None:
        self._window = window
        self._season_length = season_length

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Extract the latest close date for each symbol
        latest_closes = view.closes(lookback=self._window).select(
            pl.all().exclude("session_date")
        )

        # Filter out symbols not present in history
        valid_symbols = set(view.symbols) & set(latest_closes.columns)

        if len(valid_symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        # Compute the mean returns for each symbol over the season length
        means: dict[str, float] = {}
        for symbol in valid_symbols:
            closes = latest_closes[symbol].to_list()
            if len(closes) < self._season_length:
                continue

            start_date = history.filter(
                (pl.col("session_date") >= stamp - pl.duration(days=self._season_length))
                & (pl.col("session_date") <= stamp)
            )["close"].mean()

            end_date = history.filter(
                (pl.col("session_date") > stamp - pl.duration(days=self._window))
                & (pl.col("session_date") < stamp)
            )["close"].mean()

            mean_return = (end_date / start_date) - 1.0
            means[symbol] = float(mean_return)

        # Rank symbols by their mean returns in descending order
        ranked_symbols = [
            symbol for _, symbol in sorted(means.items(), key=lambda item: item[1], reverse=True)
        ]

        top_n_symbols = ranked_symbols[:3]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest