from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price levels revert to the mean over time. By identifying symbols that have "
        "deviated significantly from their historical average price and then trading back towards it, "
        "we can exploit this reversion tendency for profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_price = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns(
                (pl.col("close") / pl.col("mean") - 1.0).alias("deviation")
            )
        )

        symbols_with_deviation = mean_price.select(
            ["symbol", "deviation"]
        ).to_pandas()

        top_symbols = (
            symbols_with_deviation.sort_values(by="deviation", ascending=False)
            .head(int(self._threshold * len(symbols_with_deviation)))
            .set_index("symbol")["deviation"].to_dict()
        )

        bottom_symbols = (
            symbols_with_deviation.sort_values(by="deviation")
            .head(int(self._threshold * len(symbols_with_deviation)))
            .set_index("symbol")["deviation"].to_dict()
        )

        weights: dict[str, float] = {}
        for symbol in top_symbols:
            if symbol in view.symbols and history.select(["symbol", "adj_close"]).filter(pl.col("symbol") == symbol).height >= self._window:
                weights[symbol] = -top_symbols[symbol]
        
        for symbol in bottom_symbols:
            if symbol in view.symbols and history.select(["symbol", "adj_close"]).filter(pl.col("symbol") == symbol).height >= self._window:
                weights[symbol] = -bottom_symbols[symbol]

        if not weights:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        return Signal(
            information_available_at=stamp,
            weights={s: abs(normalized_weights[s]) for s in view.symbols if s in normalized_weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest