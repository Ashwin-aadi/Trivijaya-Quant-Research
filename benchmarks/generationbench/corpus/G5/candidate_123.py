from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for the most liquid stocks in the NIFTY 100 by looking at "
        "volume. It then equal weights these stocks to exploit liquidity and reduce transaction costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = history.select(
            pl.col("symbol"), pl.col("volume").sum().alias("total_volume")
        )
        sorted_volume = volume_df.sort("total_volume", descending=True).head(50)
        
        if sorted_volume.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in sorted_volume.iter_rows()]
        weight = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest