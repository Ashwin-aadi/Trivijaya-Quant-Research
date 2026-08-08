from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on identifying and investing in a portfolio of highly liquid "
        "stocks with equal weighting. High liquidity can indicate market confidence and lower "
        "transaction costs, while equally weighting stocks mitigates concentration risk."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        hist = view.history(lookback=self._window)
        if hist.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily turnover ratio
        hist = (
            hist
            .with_column((pl.col("volume") / (pl.col("adj_close") * pl.col("volume").sum() / pl.col("volume").count())).alias("turnover_ratio"))
            .sort("session_date", descending=True)
            .head(self._window)
        )

        # Rank stocks by turnover ratio
        ranked = hist.group_by("symbol").agg(pl.col("turnover_ratio").mean().alias("avg_turnover_ratio")).sort("avg_turnover_ratio", descending=True)

        if ranked.height < 30:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in ranked.to_dicts()[:30]]

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