from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion60d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that have "
        "deviated significantly from their historical price levels. By entering positions when "
        "stocks are undervalued relative to their historical norms, we aim to profit from "
        "the tendency for prices to revert to the mean."
    )

    def __init__(self, lookback_days: int = 60, z_score_threshold: float = -2) -> None:
        self._lookback_days = lookback_days
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_days)
        if closes.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._lookback_days)
            history = history.sort("session_date").filter(pl.col("symbol") == symbol).select(
                pl.all().exclude("symbol")
            )
            prices = [float(v) for v in history["adj_close"].to_list()]
            mean_price = sum(prices) / len(prices)
            std_dev = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5
            z_score = (closes[symbol].max() - mean_price) / std_dev

            if z_score < self._z_score_threshold:
                picks.append(symbol)

        picks = sorted(picks, key=lambda s: closes[s].max(), reverse=True)[:20]
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