from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion6m(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that have "
        "deviated significantly from their historical price trends. High volatility and recent "
        "price deviations suggest temporary market inefficiencies, offering opportunities for "
        "profit through short positions."
    )

    def __init__(self, window: int = 180, threshold: float = 2.0, max_positions: int = 20) -> None:
        self._window = window
        self._threshold = threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].unique().to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        z_scores = []
        for symbol in symbols:
            close_prices = history.filter(pl.col("symbol") == symbol)["adj_close"]
            mean_price = close_prices.mean()
            std_deviation = close_prices.std()

            daily_z_score = (history.filter(pl.col("symbol") == symbol)["adj_close"] - mean_price) / std_deviation
            z_scores.append((symbol, daily_z_score.max().item()))

        ranked_stocks = sorted(z_scores, key=lambda x: abs(x[1]), reverse=True)[: self._max_positions]
        weights = {stock[0]: 0.05 for stock in ranked_stocks}

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in ranked_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).item()
    assert isinstance(newest, date)
    return newest