from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation20d(Strategy):
    rationale = (
        "This strategy targets stocks that have recently broken through significant price levels "
        "and aims to capture the continuation phase of the breakout trend. It leverages momentum "
        "and volume around breakouts to filter out weaker signals and enter long positions."
    )

    def __init__(self, window: int = 20, retracement_threshold: float = -0.05, max_positions: int = 20) -> None:
        self._window = window
        self._retracement_threshold = retracement_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate rolling high and low
        history = (
            history
            .with_columns(
                (pl.col("close").rolling_max(window=self._window).alias(f"20d_high")).shift(1),
                (pl.col("close").rolling_min(window=self._window).alias(f"20d_low")).shift(1)
            )
            .filter(pl.col("session_date") < view.as_of)
        )

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify breakouts
        breakout_symbols = []
        for symbol in view.symbols:
            recent_close = [float(v) for v in history.select(symbol).to_series().drop_nulls().to_list()]
            if len(recent_close) < self._window + 1 or recent_close[-1] <= recent_close[-self._window - 1]:
                continue

            # Check if the last close price is a breakout
            if (recent_close[-1] > history.select(f"20d_high")[symbol][-1]) or \
               (recent_close[-1] < history.select(f"20d_low")[symbol][-1]):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Rank and select top N breakouts
        ranked_breakouts = sorted(breakout_symbols,
                                  key=lambda s: (history.select("session_date").filter(pl.col(s) == history.select(f"20d_high")[s][-1]).item(),  # Last session of breakout
                                                 -history.select(f"20d_low")[s].to_list().index(recent_close[-self._window])))

        selected_symbols = ranked_breakouts[:self._max_positions]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Define entry and holding criteria
        weight = 1.0 / len(selected_symbols)
        signal_weights = {s: weight for s in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest