from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often signals of significant underlying "
        "market sentiment. By combining price movements with volume changes, we can identify "
        "trading opportunities that may be more reliable."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and volume changes
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
                (pl.col("volume") - pl.col("volume").shift(1)).alias("v_change")
            )
            .sort("session_date", descending=False)
        )

        # Filter symbols based on non-zero returns and significant volume changes
        filtered_symbols = [
            symbol for symbol in view.symbols
            if history.select(
                (pl.col("r") * pl.col("v_change")).alias(f"{symbol}_imp")
            ).filter(pl.col(f"{symbol}_imp").abs() > 0.1).height > 0
        ]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest