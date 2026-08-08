from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can "
        "potentially lead to continuation of the trend. By focusing on symbols that show"
        " significant volume increase alongside price movement, we aim to capitalize on "
        "these high-probability trends."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volatility_symbols = (
            history.select(
                pl.col("symbol"), (pl.col("volume") / pl.col("adj_close").shift(-1)).alias("vol_ratio")
            )
            .with_columns((pl.col("close") - pl.col("open")).abs().alias("price_move"))
            .filter(
                (pl.col("vol_ratio") > 2)
                & (pl.col("price_move") / pl.col("adj_close").shift(1) >= 0.5)
            )
            .group_by("symbol")
            .agg(pl.count())
            .sort("count", descending=True)
            .limit(5)
        )

        if high_volatility_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in high_volatility_symbols.to_dicts()]
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