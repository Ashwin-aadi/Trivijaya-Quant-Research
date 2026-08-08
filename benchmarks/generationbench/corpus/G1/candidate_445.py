from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion exploits the tendency of financial assets to return towards their "
        "historical average price over time. This strategy identifies symbols that have deviated"
        " significantly from their mean and bets on a return towards this mean."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.melt().group_by("symbol").agg(pl.col("value").mean()).collect()["value"]
        ).to_list()

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            z_score = (latest_close - mean_close[0]) / (mean_close[1] if mean_close[1] != 0 else 1)
            if abs(z_score) > self._threshold:
                picks.append(symbol)

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