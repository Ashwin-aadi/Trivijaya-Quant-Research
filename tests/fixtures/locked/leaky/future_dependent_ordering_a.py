"""Quality-compounder momentum basket for NIFTY 100 constituents.

Restricts trading to a small set of names that have proven to be India's most durable
compounders, then rotates within that set using short-term momentum. Concentrating on a
pre-vetted list of resilient franchises is meant to avoid the whipsaw that comes from chasing
momentum in names whose businesses are not built to sustain a multi-year rally.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class QualityCompounderMomentum(Strategy):
    """Rotates by trailing momentum, restricted to a fixed set of proven compounders."""

    rationale = (
        "Momentum rotation works best when confined to businesses durable enough to sustain a "
        "multi-year uptrend, rather than being chased in every name that has one good month. "
        "Identifying which franchises actually compounded over the full study period lets the "
        "strategy avoid rotating into names whose apparent trend is really just noise."
    )

    def __init__(
        self, universe_history: pl.DataFrame, basket_size: int = 15, top_n: int = 5
    ) -> None:
        self._top_n = top_n
        self._core_basket = self._select_compounders(universe_history, basket_size)

    @staticmethod
    def _select_compounders(panel: pl.DataFrame, basket_size: int) -> tuple[str, ...]:
        total_return = (pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias(
            "total_return"
        )
        ranked = (
            panel.sort("session_date")
            .group_by("symbol")
            .agg(total_return)
            .sort("total_return", descending=True)
            .head(basket_size)
        )
        return tuple(ranked["symbol"].to_list())

    def generate(self, view: MarketView) -> Signal:
        history = view.history(63)
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        eligible = [s for s in self._core_basket if s in view.symbols]
        momentum = (
            history.filter(pl.col("symbol").is_in(eligible))
            .group_by("symbol")
            .agg((pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias("mom"))
            .sort("mom", descending=True)
            .head(self._top_n)
        )
        symbols = momentum["symbol"].to_list()
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols} if symbols else {}
        return Signal(information_available_at=view.as_of, weights=weights)
