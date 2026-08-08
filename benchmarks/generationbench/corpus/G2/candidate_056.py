from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that a significant amount of buying or "
        "selling pressure has been realized. This can indicate a strong trend and potentially "
        "lead to continuation of the move."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.height < self._window + 30:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and volume
        returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
                "volume",
            )
            .sort("session_date")
            .drop_nulls()
            .group_by("symbol", maintain_order=True)
            .agg(
                (
                    (pl.col("r").sum().alias("total_return")),
                    (pl.col("volume").max().alias("max_volume")),
                )
            )
        )

        # Find symbols with significant moves and high volume
        breakout_symbols = []
        for symbol in view.symbols:
            if returns.get_column(symbol).is_empty():
                continue

            total_return = float(returns.get_column(symbol)["total_return"][0])
            max_volume = float(returns.get_column(symbol)["max_volume"][0])

            if abs(total_return) >= self._threshold and max_volume > 0:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest