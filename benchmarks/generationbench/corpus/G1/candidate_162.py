from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broad universe "
        "helps to identify outperformers. This strategy focuses on relative performance "
        "to filter for potentially strong stocks."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        broad_universe_strength = (
            closes
            .select([pl.col(col).rank(method="dense", descending=True).alias(f"r_{col}")
                     for col in view.symbols])
            .agg_over(["r_" + symbol for symbol in view.symbols], pl.min())
            .to_dict(True)
        )

        symbols_strength = {symbol: strength for symbol, strength in broad_universe_strength["r"].items() if strength != 1}

        top_n_symbols = sorted(symbols_strength.items(), key=lambda x: x[1])[:5]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest