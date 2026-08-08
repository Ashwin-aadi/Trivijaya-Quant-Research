from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which are far from their historical "
        "averages will tend to move back towards them over time. By identifying stocks "
        "that have moved too far in one direction and betting on a return to the mean, we "
        "can profit."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("mean"))
        )
        latest_closes = view.closes()
        
        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_close["symbol"].to_list() or symbol not in latest_closes.columns:
                continue
            mean_value = float(mean_close.filter(pl.col("symbol") == symbol)["mean"][0])
            recent_close = float(latest_closes[symbol].item())
            
            z_score = (recent_close - mean_value) / mean_value
            
            if abs(z_score) > 2.0:  # Consider using a different threshold
                signals[symbol] = -1.0 * z_score

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        
        total_weight = sum(signals.values())
        adjusted_weights = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, 
            weights=adjusted_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest