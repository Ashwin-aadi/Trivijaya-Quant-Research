from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the recent momentum of a stock with its volatility. "
        "High momentum suggests strong price movement in one direction, while low volatility "
        "indicates that the price is stable and less likely to reverse."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.is_empty() or history.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            price_changes = (
                (history.select(pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0))
                .select((pl.all().mean()).alias("avg_change"))
                .get_column("avg_change")
                .to_list()
            )
            if len(price_changes) < self._momentum_window:
                continue

            momentum_scores[symbol] = abs(sum(price_changes[-self._momentum_window:]))

            vol_data = history.select(pl.col("adj_close").std())
            volatility = float(vol_data.to_series().item())
            volatility_scores[symbol] = volatility

        combined_scores = {
            symbol: (momentum_scores.get(symbol, 0) / self._momentum_window + volatility_scores.get(symbol, 0)) 
            for symbol in momentum_scores.keys() & volatility_scores.keys()
        }

        top_symbols = sorted(combined_scores.items(), key=lambda x: -x[1])[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().to_series().item()
    assert isinstance(newest, date)
    return newest