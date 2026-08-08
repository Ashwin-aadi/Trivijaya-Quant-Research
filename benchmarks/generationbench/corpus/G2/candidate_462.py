from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Firms with strong relative strength are more likely to outperform the broader market "
        "as they may be better managed or have a competitive edge. This strategy identifies and "
        "invests in symbols that have been outperforming their peers over a recent period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        returns = (closes[symbols].to_numpy() / closes[symbols].shift(1).to_numpy() - 1.0).T
        avg_returns = pl.DataFrame(returns.mean(axis=1)).with_columns(
            (pl.col(0) > history["adj_close"].mean()).alias("outperform")
        )
        outperforming_symbols = symbols[avg_returns.select(pl.col("outperform")).to_list()[0]]

        if not outperforming_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(outperforming_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in outperforming_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest