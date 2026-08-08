from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a security's price deviates significantly from its mean. "
        "By identifying those that have deviated the most and are likely to revert towards their mean, we can generate profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_adj_close = (
            closes.groupby("symbol").agg(pl.col("adj_close").mean().alias("m")).select(["symbol", "m"])
        )
        latest_closes = view.closes()
        
        reversion_scores: list[float] = []
        picks: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in mean_adj_close["symbol"].to_list():
                continue
            adj_close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            latest_adj_close_value = float(latest_closes[0][symbol])
            mean_adj_close_value = mean_adj_close.filter(pl.col("symbol") == symbol)["m"].item()
            score = abs(latest_adj_close_value - mean_adj_close_value)
            reversion_scores.append(score)

        top_n_scores = sorted(reversion_scores, reverse=True)[: self._window]
        
        for i, symbol in enumerate(view.symbols):
            if symbol not in closes.columns or symbol not in mean_adj_close["symbol"].to_list():
                continue
            score = abs(
                float(latest_closes[0][symbol]) - 
                mean_adj_close.filter(pl.col("symbol") == symbol)["m"].item()
            )
            if score >= top_n_scores[self._window // 2]:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest