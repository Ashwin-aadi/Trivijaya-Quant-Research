from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards its recent mean. By using a trailing window, we capture "
        "the recent price action and adjust our view on the asset's fair value."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or (history.height - 1) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate trailing mean
        trailing_mean_adj_close = (
            history.select(pl.col("adj_close").mean())
            .to_series()
            .item()
        )
        
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(closes[symbol].max().item())
            spread = (latest_close - trailing_mean_adj_close) / trailing_mean_adj_close

            # Calculate the score based on how far away from the mean the latest close is
            signal_strength = abs(spread)
            signals[symbol] = signal_strength

        sorted_signals = {k: v for k, v in sorted(signals.items(), key=lambda item: item[1], reverse=True)}
        
        top_symbols = list(sorted_signals.keys())[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().item()
    assert isinstance(newest, date)
    return newest