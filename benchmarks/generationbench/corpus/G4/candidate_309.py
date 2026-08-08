from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical evidence that low-volatility stocks tend to "
        "outperform high-volatility stocks over long periods. By systematically allocating "
        "capital towards low-volatility equities, we aim to reduce overall portfolio risk and "
        "enhance returns through a diversified approach that capitalizes on mispriced securities."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day realized volatility for each stock
        history = (
            history.with_columns(
                (pl.col("close") - pl.col("open")).abs().mean().over(pl.col("symbol")) * 2.5066
                .alias(f"volatility")
            )
            .sort("session_date", descending=True)
            .with_column((pl.col(f"volatility").rank(method="dense", descending=True)).alias("rank"))
        )

        low_vol_symbols = history.filter(pl.col("rank") < self._top_n + 1)["symbol"].to_list()

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest