from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum. "
        "By identifying symbols that show a significant price move with high volume, we can capitalize on trending behavior."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.select(
                [
                    pl.col("symbol"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                    pl.col("volume"),
                ]
            )
            .sort("session_date")
            .with_column((pl.col("return") * pl.col("volume")).alias("volatility_return"))
        )

        # Filter out symbols with too little history
        filtered_history = returns.filter(pl.col("symbol").is_in(view.symbols)).group_by(
            "symbol"
        ).agg(
            (
                (pl.col("return").mean().alias("avg_return")),
                (pl.col("volatility_return").mean().alias("avg_vol_return")),
            )
        )

        # Identify symbols with significant directional moves
        breakout_symbols = filtered_history.filter(
            pl.col("avg_return") > 0.01 * filtered_history.select(pl.col("avg_vol_return").max())
        ).select(["symbol"])

        if breakout_symbols.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())
    assert isinstance(newest.item(), date)
    return newest.item()