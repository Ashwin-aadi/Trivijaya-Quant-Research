from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves identifying assets that are trending "
        "in a particular direction and scaling the position size based on recent volatility. "
        "This strategy aims to capture trends while mitigating risk by adjusting exposure."
    )

    def __init__(self, window: int = 20, trend_factor: float = 1.5) -> None:
        self._window = window
        self._trend_factor = trend_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = [float(v) for v in view.closes(lookback=self._window).to_dict(False)[1].values()]
        prices = pl.DataFrame({"symbol": symbols, "close": closes})
        price_changes = prices.with_columns(
            (pl.col("close").shift(-1) - pl.col("close")) / pl.col("close") * 100.0
        ).sort("symbol")
        trend_scores = (
            price_changes.select(
                pl.col("symbol"),
                ((pl.col("close").rolling_sum(2) - pl.col("close").shift(1).rolling_sum(2)) / self._window)
                .abs()
                .rank(method="dense", descending=True)
                .alias("trend_score")
            )
        ).collect()

        weighted_scores = trend_scores.with_columns(
            (pl.col("trend_score") * self._trend_factor).alias("weighted_score")
        )

        top_symbols = [row["symbol"] for row in weighted_scores.sort("weighted_score", descending=True).limit(5)]
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest