from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "This strategy aims to capture price reversions around historical support and resistance "
        "levels in Indian equities. By identifying key levels and exploiting mean reversion, the "
        "strategy seeks to capitalize on temporary market imbalances and historical price patterns."
    )

    def __init__(self, window: int = 200, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        support_resistance = self._identify_support_resistance(history)
        if not support_resistance:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            current_close = float(view.latest_close()[symbol])
            deviations = {level: abs(current_close - level) for level in support_resistance[symbol]}
            top_levels = sorted(deviations.items(), key=lambda x: (x[1], -deviations[x[0]]))[: self._top_n]
            for level, deviation in top_levels:
                weight = 1.0 / len(top_levels)
                signals[symbol] = max(0.0, weight) if current_close < level else 0.0

        return Signal(information_available_at=stamp, weights={s: w for s, w in signals.items() if w > 0})

    def _identify_support_resistance(self, history: pl.DataFrame) -> dict[str, list[float]]:
        symbols = history["symbol"].unique().to_list()
        support_resistance: dict[str, list[float]] = {symbol: [] for symbol in symbols}

        for i in range(1, self._window):
            windowed_history = history.slice(i - 1, self._window)
            for symbol in symbols:
                sub_df = windowed_history.filter(pl.col("symbol") == symbol)
                if sub_df.is_empty():
                    continue
                min_price = float(sub_df["adj_close"].min())
                max_price = float(sub_df["adj_close"].max())
                support_resistance[symbol].append(min_price)
                support_resistance[symbol].append(max_price)

        return {k: sorted(list(set(v))) for k, v in support_resistance.items() if len(v) > 0}

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest