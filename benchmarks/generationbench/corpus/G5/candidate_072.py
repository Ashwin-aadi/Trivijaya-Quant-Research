from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to the market over a certain period to continue outperforming in the future."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(2 * self._lookback)
        )

        # Get top performers relative to the market
        market_return = history_with_returns.select(
            pl.col("adj_close").shift(-self._lookback) / view.latest_close()[list(view.symbols)[0]] - 1.0
        ).select(pl.col("return")).mean().item()
        excess_returns = history_with_returns.with_columns(
            (pl.col("return") - market_return).alias("excess_return")
        )

        # Rank symbols by excess return
        ranked = (
            excess_returns.group_by("symbol")
            .agg(
                pl.col("excess_return").mean().rank(method="ordinal", descending=True).alias("rank")
            )
            .sort("rank")
        )

        top_symbols = [row["symbol"] for row in ranked.head(self._lookback + 1).to_dicts()]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest