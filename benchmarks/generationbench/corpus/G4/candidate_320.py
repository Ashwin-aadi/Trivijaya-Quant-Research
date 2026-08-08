from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large trading volumes often precede significant price movements, indicating strong "
        "institutional or high-frequency trader interest. This strategy aims to identify and "
        "enter into positions where volume spikes are confirmed by subsequent directional price "
        "movements."
    )

    def __init__(self, lookback_days: int = 30, min_volume_change: float = 20.0, min_price_move: float = 0.01, top_n: int = 10) -> None:
        self._lookback_days = lookback_days
        self._min_volume_change = min_volume_change
        self._min_price_move = min_price_move
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily volume change percentage
        history = (
            history
            .with_columns(
                (pl.col("volume") / pl.col("volume").shift_or_null(1) - 1.0).alias("volume_change")
            )
            .sort("session_date", descending=True)
        )

        # Filter recent sessions where volume change is significant and price move is in the same direction
        significant_moves = (
            history
            .filter(
                (pl.col("volume_change") >= self._min_volume_change) &
                ((pl.col("close") / pl.col("adj_close").shift(1) - 1.0) * pl.col("volume_change").abs() > self._min_price_move)
            )
        )

        # Rank symbols by volume change
        ranked = significant_moves.group_by("symbol").agg(
            (pl.col("volume_change") / pl.col("volume_change").max().over()).alias("rank")
        ).sort("rank", descending=True)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in ranked.columns or len(ranked[symbol].to_list()) < self._lookback_days + 1:
                continue
            rank_value = float(ranked[symbol][-2])
            if rank_value > 0.0:  # Consider only symbols with non-zero rank value for simplicity
                picks.append(symbol)

        picks = picks[:self._top_n]
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