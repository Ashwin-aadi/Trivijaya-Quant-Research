from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeFactorStrategy(Strategy):
    rationale = (
        "This strategy combines market capitalization (size) with earnings quality to "
        "create a multifactor model. Small-cap stocks often exhibit higher growth potential, "
        "while earnings quality helps filter out unreliable financials, providing a balanced approach."
    )

    def __init__(self, size_weight: float = 0.7, quality_weight: float = 0.3) -> None:
        self._size_weight = size_weight
        self._quality_weight = quality_weight

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=3 * 252)  # 3 years of daily data
        if history.height < 756:  # Less than 3 years is too short for meaningful analysis
            return Signal(information_available_at=stamp, weights={})

        size_factor = self._size_score(history)
        quality_factor = self._quality_score(history)

        composite_scores = (size_factor * self._size_weight) + (quality_factor * self._quality_weight)
        ranked_stocks = composite_scores.argsort(descending=True).to_list()

        top_n = 30
        if len(ranked_stocks) < top_n:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = [view.symbols[i] for i in ranked_stocks[:top_n]]
        bottom_stocks = [view.symbols[i] for i in ranked_stocks[-top_n:]]

        long_weights = {s: 1.0 / len(top_stocks) for s in top_stocks}
        short_weights = {s: -1.0 / len(bottom_stocks) for s in bottom_stocks}

        return Signal(
            information_available_at=stamp,
            weights={**long_weights, **short_weights},
        )

    def _size_score(self, history: pl.DataFrame) -> pl.Series:
        market_cap = self._latest_market_cap(history)
        log_mkt_cap = (pl.col("adj_close") * 1e-6).log()  # Assuming MC is in million
        z_scores = (log_mkt_cap - log_mkt_cap.mean()) / log_mkt_cap.std()
        return z_scores

    def _quality_score(self, history: pl.DataFrame) -> pl.Series:
        margin_metrics = self._margin_metrics(history)
        quality_z_scores = (
            (pl.col("operating_margin") + pl.col("net_profit_margin") + pl.col("roa")) / 3
        ).rank(method="ordinal", descending=True).to_frame().with_columns((1.0 - pl.arange(1, name="z_score").cast(pl.Float64)) / len(history))
        return quality_z_scores["z_score"]

    def _latest_market_cap(self, history: pl.DataFrame) -> pl.Series:
        latest_close = view.latest_close()
        market_cap = (pl.col("adj_close") * 1e-6).alias("market_cap")
        return market_cap.with_columns(market_cap.max().over("symbol").alias("max_mkt_cap"))

    def _margin_metrics(self, history: pl.DataFrame) -> pl.DataFrame:
        operating_margin = (history["close"] - history["cost_of_goods_sold"]) / history["close"]
        net_profit_margin = history["net_income"] / history["close"]
        roa = history["net_income"] / history["total_assets"]
        return history.select(["symbol", "session_date", "close", "net_income", "cost_of_goods_sold", "total_assets", operating_margin, net_profit_margin, roa])


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest