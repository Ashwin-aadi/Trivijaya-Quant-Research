from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broad market "
        "tends to outperform in the long run. This is based on the idea that strong "
        "sectors or themes may continue to perform well."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        market_close = float(view.latest_close()["NIFTY 100"].max())
        closes = pl.DataFrame(history).select(pl.col("symbol").alias("Symbol"), "close")

        symbol_strengths = (
            closes
            .group_by("Symbol")
            .agg(
                (pl.col("close") / pl.col("close").shift(self._window - 1) - 1.0)
                .mean()
                .alias("strength"),
                (pl.col("close") / market_close).mean().alias("market_strength"),
            )
        ).sort("strength", descending=True)

        if symbol_strengths.height < view.symbols.count():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(row.Symbol) for row in symbol_strengths.to_dicts()][:10]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest