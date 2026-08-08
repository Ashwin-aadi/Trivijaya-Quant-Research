from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with high returns relative to their "
        "peers over a recent period are likely to continue outperforming. This strategy aims to "
        "identify such stocks and allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        latest_closes = {s: float(v) for s, v in view.latest_close().items() if s in symbols}
        
        # Calculate returns over the lookback period
        returns_df = (
            history
            .select([pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")])
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("mean_return"))
            .sort("mean_return", descending=True)
        )

        top_symbols = [s for s in returns_df.select("symbol").to_list()[0][:5] if s in symbols]
        
        # Ensure at least one symbol is selected
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