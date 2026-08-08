from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "Historical price data indicates that extreme deviations from a long-term mean are "
        "temporarily sustainable but eventually revert. This strategy leverages Z-scores to "
        "identify such opportunities and rebalance the portfolio accordingly."
    )

    def __init__(self, window: int = 50, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_price = pl.Series(closes).mean().item()
        std_dev = pl.Series(closes).std().item()

        z_scores: list[float] = []
        for i in range(self._window, len(closes)):
            z_score = (closes[i] - mean_price) / std_dev
            z_scores.append(z_score)

        picks: list[str] = [view.symbols[i + self._window] for i, score in enumerate(z_scores) if abs(score) > 2]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest