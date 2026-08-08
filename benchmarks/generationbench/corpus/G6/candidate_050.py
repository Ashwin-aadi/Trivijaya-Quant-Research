from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "The strategy aims to track stock price movements while adjusting sensitivity based on recent volatility, "
        "ensuring enhanced risk management during high-volatility periods. By using exponentially weighted moving average (EWMA) for momentum calculation and dynamic entry/exit rules, we aim to capture trends effectively while limiting potential losses."
    )

    def __init__(self, window: int = 50, ewma_window: int = 20, max_positions: int = 30) -> None:
        self._window = window
        self._ewma_window = ewma_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        ewma_close = (history["close"] / history["adj_close"]).ewm(span=self._ewma_window).mean()
        recent_volatility = (history["high"] - history["low"]).std()

        symbol_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in ewma_close.columns or symbol not in recent_volatility.columns:
                continue
            ewma_score = ewma_close[symbol][-1]
            vol_score = 1.0 / (recent_volatility[symbol] + 1e-6)
            total_score = ewma_score * vol_score
            symbol_scores[symbol] = total_score

        sorted_symbols = [k for k, v in sorted(symbol_scores.items(), key=lambda item: -item[1])]
        selected_symbols = sorted_symbols[: self._max_positions]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest