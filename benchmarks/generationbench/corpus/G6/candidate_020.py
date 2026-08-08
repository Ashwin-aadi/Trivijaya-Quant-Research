from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBackedMove(Strategy):
    rationale = (
        "This strategy identifies stocks showing strong volume-backed directional moves to capture momentum and liquidity. "
        "It enters positions when a stock's closing price significantly deviates from its opening price with substantial trading volume, "
        "ensuring alignment between price and volume signals."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5, profit_target: float = 0.25, loss_limit: float = 0.1) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio
        self._profit_target = profit_target
        self._loss_limit = loss_limit

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        filtered_history = history.select(pl.col("symbol").is_in(filtered_symbols))

        price_moves = (
            filtered_history
            .group_by("symbol")
            .agg(
                (pl.col("close") - pl.col("open")).abs().max().alias("price_move"),
                (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("volume_ratio"),
                (pl.col("close") - pl.col("open")) * (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("direction")
            )
        )

        picks: list[str] = []
        for symbol in filtered_symbols:
            row = price_moves.filter(pl.col("symbol") == symbol).row(0)
            if not row:
                continue
            price_move, volume_ratio, direction = row[1], float(row[2]), int(row[3])
            if (
                price_move > 0
                and volume_ratio >= self._min_volume_ratio
                and direction == 1
            ):
                picks.append(symbol)

        picks = sorted(picks)[:20]
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