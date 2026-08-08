from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of their initial move. Identifying "
        "breakouts that maintain momentum can provide profitable trading signals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            open_price = prices[0]
            close_price = prices[-1]

            # Calculate the breakout condition
            if close_price > open_price * 1.02:  # 2% above open price
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Filter to find symbols that continue in the direction of their initial move
        continuation_symbols = [
            symbol for symbol in breakout_symbols if _continuation_check(view, symbol)
        ]

        weight = 1.0 / len(continuation_symbols) if continuation_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _continuation_check(view: MarketView, symbol: str) -> bool:
    history = view.history(lookback=view._window + 20)
    if history.is_empty() or history.height < view._window + 20:
        return False

    # Check the continuation condition
    initial_move_direction = _initial_move_direction(view, symbol)
    recent_prices = [
        float(v) for v in history[symbol].sort("session_date").to_list()
    ]
    for i in range(1, len(recent_prices)):
        if (recent_prices[i] - recent_prices[i - 1]) * initial_move_direction < 0:
            return False
    return True


def _initial_move_direction(view: MarketView, symbol: str) -> int:
    history = view.history(lookback=view._window + 20)
    if history.is_empty() or history.height < view._window + 20:
        return 0

    prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
    open_price = prices[0]
    close_price = prices[-1]

    return int((close_price - open_price) / abs(close_price - open_price))