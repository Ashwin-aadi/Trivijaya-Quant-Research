from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum can reduce the noise in a single "
        "momentum signal while maintaining the potential for return. Short-term momentum "
        "signals rapid price movements, whereas long-term momentum signals sustained "
        "trends over an extended period."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes_short = (history["adj_close"].to_list()[-self._short_window:] / history["adj_close"].shift(1).drop_nulls().to_list()[:-1] - 1.0)
        closes_long = (history["adj_close"].to_list()[-self._long_window:] / history["adj_close"].shift(1).drop_nulls().to_list()[:-self._long_window] - 1.0)

        momentum_short = sum([float(v) for v in closes_short])
        momentum_long = sum([float(v) for v in closes_long])

        if momentum_short > 0 and momentum_long > 0:
            top_symbols: list[str] = []
            symbols_with_data = [symbol for symbol in view.symbols if symbol in history.columns]
            for symbol in symbols_with_data:
                if symbol not in history.symbol.to_list():
                    continue
                short_momentum = sum([float(v) for v in (history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-self._short_window:] / history.filter(pl.col("symbol") == symbol)["adj_close"].shift(1).drop_nulls().to_list()[:-1] - 1.0)])
                long_momentum = sum([float(v) for v in (history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-self._long_window:] / history.filter(pl.col("symbol") == symbol)["adj_close"].shift(1).drop_nulls().to_list()[:-self._long_window] - 1.0)])
                if short_momentum > 0 and long_momentum > 0:
                    top_symbols.append(symbol)

            weight = 1.0 / len(top_symbols)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in top_symbols}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest