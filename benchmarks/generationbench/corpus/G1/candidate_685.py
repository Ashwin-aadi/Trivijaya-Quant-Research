from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of stock prices to revert "
        "to their historical average. This strategy buys stocks that have fallen far from "
        "their recent mean and sells those that have risen too much."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            closes
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean_price")))
            .select(["symbol", "mean_price"])
            .to_dict(False)
        )

        symbols_and_scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if f"{symbol}_price" not in mean_prices:
                continue
            latest_close = view.latest_close()[symbol]
            mean_price = mean_prices[f"{symbol}_price"]
            z_score = (latest_close - mean_price) / pl.col("adj_close").std().over("symbol")
            symbols_and_scores.append((symbol, float(z_score)))

        sorted_scores = sorted(symbols_and_scores, key=lambda x: abs(x[1]), reverse=True)
        top_buyers = [s for s, _ in sorted_scores[:5] if _ < -2.0]
        top_sellers = [s for s, _ in sorted_scores[-5:] if _ > 2.0]

        weights: dict[str, float] = {}
        if top_buyers:
            weight_per_buyer = 1.0 / len(top_buyers)
            for symbol in top_buyers:
                weights[symbol] = weight_per_buyer
        if top_sellers:
            weight_per_seller = -1.0 / len(top_sellers)
            for symbol in top_sellers:
                weights[symbol] = weight_per_seller

        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest