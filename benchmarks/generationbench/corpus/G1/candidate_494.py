from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends by identifying symbols with the highest volatility-adjusted "
        "price changes. Higher volatility suggests that recent price movements are more significant, and thus potentially more profitable."
    )

    def __init__(self, window: int = 20, k: float = 1.0) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_adjusted_changes: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_values = [float(v) for v in history[symbol][0].to_list()]
            close_values = [float(v) for v in history[symbol][-1].to_list()]
            log_returns = [
                (close / open - 1.0) if open != 0 else 0.0
                for open, close in zip(open_values[:-1], close_values[1:])
            ]
            mean_log_return = sum(log_returns) / len(log_returns)
            volatility = pl.DataFrame({"log_returns": log_returns})["log_returns"].std().to_list()[0]
            if not isinstance(volatility, (int, float)):
                continue
            adjusted_change = self._k * (mean_log_return - mean_log_return.mean())
            volatility_adjusted_changes[symbol] = abs(adjusted_change) / volatility

        sorted_changes = sorted(
            volatility_adjusted_changes.items(), key=lambda x: x[1], reverse=True
        )
        top_symbols = [symbol for symbol, _ in sorted_changes][:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest