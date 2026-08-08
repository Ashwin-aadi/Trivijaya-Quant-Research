from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their "
        "20-day simple moving average (SMA) and expects them to revert. It balances "
        "short-term responsiveness with historical context by focusing on recent "
        "underperformance or outperformance."
    )

    def __init__(self, window: int = 20, threshold_deviation: float = 5.0,
                 std_dev_threshold: float = 1.0, stop_loss: float = -10.0) -> None:
        self._window = window
        self._threshold_deviation = threshold_deviation
        self._std_dev_threshold = std_dev_threshold
        self._stop_loss = stop_loss

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma = history.groupby("symbol").agg(
            (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("sma")
        ).with_columns((pl.col("close").mean().over("symbol")).alias("20d_sma"))
        sma = sma.join(history, on="symbol", how="inner")

        # Calculate deviations from the SMA
        sma = sma.with_column(
            (pl.col("close") - pl.col(f"20d_sma")) / pl.col(f"20d_sma").shift(1).alias("deviation")
        )
        sma = sma.with_column(
            (pl.col("deviation").abs() > self._threshold_deviation) & (
                pl.col("deviation").std().over("symbol") > self._std_dev_threshold
            ).alias("signal")
        )

        # Identify candidates for reversion
        picks: list[str] = sma.filter(pl.col("signal"))["symbol"].to_list()
        if len(picks) < 50:
            picks.extend([s for s in view.symbols if s not in picks])
            picks = picks[:50]

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest