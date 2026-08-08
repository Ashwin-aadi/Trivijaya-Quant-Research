from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that securities with the highest relative strength "
        "over a recent period tend to continue outperforming. This strategy identifies top-performing "
        "stocks and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute relative strength scores
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = history.select(pl.col(symbols).last().alias("close"))
        open_prices = history.select(pl.col(symbols).first().alias("open"))

        # Calculate daily returns
        returns = (closes / open_prices - 1.0).to_series().to_list()

        # Rank symbols by their mean return over the window
        rank_expr = (
            pl.DataFrame(returns, schema={"symbol": str, "return": float})
            .with_column(
                pl.col("return").mean().over(pl.col("symbol")).alias("avg_return")
            )
            .select(pl.col("symbol"), pl.col("avg_return"))
            .sort("avg_return", descending=True)
        )

        top_symbols = rank_expr.head(5)["symbol"].to_list()
        weight = 1.0 / len(top_symbols) if top_symbols else 0.0

        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest