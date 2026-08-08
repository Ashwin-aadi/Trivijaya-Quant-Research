from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often associated with higher risk-adjusted returns. By "
        "tilting towards low-volatility assets, we aim to capture these potential excess "
        "returns while reducing overall portfolio volatility."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().alias("return_abs"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0) * 100).alias("return_pct")
            )
        )

        # Calculate mean absolute returns and standard deviation for each symbol
        stats = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("return_abs").mean().alias("mean_return")),
                (pl.col("return_abs").std().alias("volatility"))
            )
        )

        # Filter out symbols with missing values
        stats = stats.filter(pl.all().not_null())

        if stats.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Sort by volatility and select top_n low-volatility symbols
        low_vol_symbols = [row["symbol"] for row in stats.sort("volatility", descending=False).rows()[:self._top_n]]

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest