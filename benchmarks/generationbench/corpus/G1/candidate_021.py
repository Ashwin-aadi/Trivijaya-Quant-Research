from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and can provide more stable returns. "
        "By tilting the portfolio towards these stocks, we aim to reduce overall risk while maintaining competitive returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volatilities = [
            pl.col("adj_close").std().alias(f"vol_{symbol}")
            for symbol in symbols
        ]
        volatility_df = (
            history.select(symbols + volatilities)
                   .group_by(symbols)
                   .agg(pl.all().mean())
        )
        
        # Rank by mean volatility and select top N stocks
        ranked_volatility = volatility_df.sort(
            pl.col("vol_").rank(method="dense", descending=False).alias("rank")
        ).select(["symbol", "vol_", "rank"])
        picks = [row["symbol"] for _, row in ranked_volatility.iter_rows() if row["rank"] <= 5]

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