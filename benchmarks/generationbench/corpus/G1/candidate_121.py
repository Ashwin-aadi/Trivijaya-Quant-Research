from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and returns tend to revert to the mean over "
        "time. For short horizons like 10 days, a security that has performed poorly in recent "
        "sessions is likely to outperform its peers in the near future."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_dict(as_series=False)["adj_close"]
        std_dev = closes.select(
            (pl.col("adj_close") - pl.lit(mean_close)).pow(2).mean()
            .sqrt()
            .alias("std")
        ).select("std").item()

        scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            score = (values[-1] - mean_close) / std_dev
            scores[symbol] = score

        sorted_scores = sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_scores[:5]]

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