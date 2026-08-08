from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur when a security that has been trending in one direction "
        "breaks out of its recent range and then continues to move in the same direction. This "
        "behavior suggests strong momentum and can be exploited for profit."
    )

    def __init__(self, breakout_window: int = 20, continuation_window: int = 10) -> None:
        self._breakout_window = breakout_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._continuation_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        breakout_signals = {s: False for s in symbols}
        
        for symbol in symbols:
            symbol_data = history.filter(pl.col("symbol") == symbol)
            
            # Identify the breakout point
            breakout_point = (
                symbol_data.sort("session_date")
                .with_column(
                    (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .group_by("symbol")
                .agg((pl.col("return").max().alias("max_return")))
                .collect()
                .get_column("max_return")
                .to_list()[0]
            )

            if breakout_point > 0:
                # Check for continuation
                continuation_data = symbol_data.filter(
                    (pl.col("session_date") > symbol_data["session_date"].tail(1)[0])
                    & (pl.col("close") / pl.col("adj_close").shift(1) - 1.0)
                    .arr.to_list()
                    .max() > 0
                )
                
                if continuation_data.height >= self._continuation_window:
                    breakout_signals[symbol] = True

        # Identify symbols that have both a breakout and subsequent continuation
        continuation_symbols = [s for s, b in breakout_signals.items() if b]
        
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest