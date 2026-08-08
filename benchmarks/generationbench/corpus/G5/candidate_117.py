from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting towards low volatility, we can potentially capture this market anomaly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").std().alias("volatility")))
            .filter(pl.col("volatility") > 0)
            .sort(by="volatility", descending=False)
        )

        if symbol_prices.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_rank = (
            symbol_prices
            .with_column(
                (pl.col("volatility").rank(method="ordinal", descending=True)).alias("rank")
            )
            .select(["symbol", "rank"])
        )

        symbols = [row["symbol"] for row in volatility_rank.to_dicts()]
        ranks = [float(row["rank"]) for row in volatility_rank.to_dicts()]

        total_rank = sum(ranks)
        weights = {s: (r / total_rank) * 1.0 for s, r in zip(symbols, ranks)}

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items() if weight > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest