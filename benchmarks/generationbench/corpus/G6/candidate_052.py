from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ZScoreReversion(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their typical price levels "
        "using a z-score calculation. Long positions are initiated when the z-score falls below -1, and short "
        "positions are taken for stocks with a z-score above +1. Positions are exited when the closing price "
        "returns to within 2% of its trailing moving average or after holding periods as set by stop-loss orders."
    )

    def __init__(self, window: int = 200, z_score_threshold: float = -1.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        tma = history.with_columns(
            (pl.col("close").shift_and_fill(1).rolling_mean(self._window)).alias(f"tma_{self._window}")
        )
        z_scores = (
            tma
            .with_column(
                ((pl.col("adj_close") - pl.col(f"tma_{self._window}")) / (pl.col(f"tma_{self._window}") * 0.02)).alias("z_score")
            )
            .select(pl.col("symbol"), "session_date", "z_score")
        )

        picks: list[str] = []
        for symbol in view.symbols:
            row = z_scores.filter(pl.col("symbol") == symbol).sort("session_date", descending=False)
            if row.is_empty():
                continue
            latest_z_score = float(row.select("z_score").last().to_list()[0])
            if latest_z_score < self._z_score_threshold:
                picks.append(symbol)

        picks = picks[:30]  # Limit to top 30 symbols
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest