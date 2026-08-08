from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to the market over a period to continue outperforming. This is based on the "
        "idea that past performance may indicate future returns."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        mean_close = (history.select("symbol", "session_date", "adj_close")
                      .group_by("symbol")
                      .agg(pl.col("adj_close").mean().alias("mean_close"))
                      .collect()
                      .select(["symbol", "mean_close"])
                      .to_dict(as_series=False))

        momentum_scores = []
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window:
                continue

            mean_adj_close = float(mean_close.get(symbol, 0))
            momentum_score = (close_series[-1] - mean_adj_close) / mean_adj_close
            momentum_scores.append((symbol, momentum_score))

        sorted_scores = sorted(momentum_scores, key=lambda x: x[1], reverse=True)
        top_symbols = [x[0] for x in sorted_scores[: self._lookback]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest