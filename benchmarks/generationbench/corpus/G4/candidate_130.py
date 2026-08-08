from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy exploits seasonal effects in the Indian market by identifying stocks "
        "that historically perform better during specific calendar events. By analyzing past "
        "price and volume data, we aim to predict and capitalize on these predictable patterns."
    )

    def __init__(self, lookback: int = 5 * 365, positive_events_window: int = 30, negative_events_window: int = 30) -> None:
        self._lookback = lookback
        self._positive_events_window = positive_events_window
        self._negative_events_window = negative_events_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify positive and negative events
        positive_events = {"festival": date(year=2024, month=10, day=5)}
        negative_events = {"monsoon": date(year=2023, month=7, day=1)}

        # Calculate signals for each stock
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue

            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            latest_close = float(view.latest_close()[symbol])

            # Calculate positive event signal
            try:
                positive_start = max(0, stamp - date(self._lookback))
                positive_returns = (df[(df["session_date"] >= positive_start) & (df["session_date"] < positive_events["festival"])][
                                        "adj_close"].to_list()[::-1])[-self._positive_events_window:]
                positive_mean_return = sum(positive_returns) / len(positive_returns)
                positive_std_dev = (sum((x - positive_mean_return)**2 for x in positive_returns) / len(positive_returns))**0.5

            except IndexError:
                continue

            # Calculate negative event signal
            try:
                negative_start = max(0, stamp - date(self._lookback))
                negative_returns = (df[(df["session_date"] >= negative_start) & (df["session_date"] < negative_events["monsoon"])][
                                        "adj_close"].to_list()[::-1])[-self._negative_events_window:]
                negative_mean_return = sum(negative_returns) / len(negative_returns)
                negative_std_dev = (sum((x - negative_mean_return)**2 for x in negative_returns) / len(negative_returns))**0.5

            except IndexError:
                continue

            # Combine signals
            positive_signal = max(positive_mean_return, 0) - min(negative_mean_return, 0)
            if positive_signal > 0:
                picks.append(symbol)

        picks = picks[:20]
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