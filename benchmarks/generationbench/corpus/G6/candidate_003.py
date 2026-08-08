from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on identifying stocks with better liquidity and equal-weighting them to minimize concentration risk. The rationale is that liquid assets are less risky and more tradable, providing smoother entry and exit opportunities."
    )

    def __init__(self, min_adtv: float = 10_000_000, window: int = 20, portfolio_size: int = 75) -> None:
        self._min_adtv = min_adtv
        self._window = window
        self._portfolio_size = portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        adtv_col_name = f"adtv_{self._window}d"
        history = (
            history
            .with_column(
                (pl.col("volume") * pl.col("close")).sum().over("symbol").alias(adtv_col_name)
            )
            .filter(pl.col(adtv_col_name) > self._min_adtv)
            .sort(adtv_col_name, descending=True)
            .head(self._portfolio_size)[["symbol"]]
        )

        symbols = history["symbol"].to_list()
        weight = 1.0 / len(symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest