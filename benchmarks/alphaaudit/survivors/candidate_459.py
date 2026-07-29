from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening is a robust way to ensure that highly traded stocks are given "
        "greater weight in the portfolio. This strategy selects the top N most liquid symbols "
        "and equal weights them within the portfolio."
    )

    def __init__(self, window: int = 20, num_assets: int = 5) -> None:
        self._window = window
        self._num_assets = num_assets

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") / pl.col("open")).mean().alias("price_change"),
            )
            .sort(by="total_volume", descending=True)
            .head(self._num_assets)[
                ["symbol", "total_volume"]
            ]  # Select top N symbols by volume
        )

        weights = {row["symbol"]: 1.0 / self._num_assets for row in liquidity.to_dicts()}
        return Signal(
            information_available_at=stamp, weights={k: float(v) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest