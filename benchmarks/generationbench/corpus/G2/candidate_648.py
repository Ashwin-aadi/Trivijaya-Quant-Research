from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the phenomenon where stocks that have performed "
        "well in the recent past tend to continue outperforming those that have underperformed. "
        "This can be attributed to various factors such as investor sentiment and market efficiency."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            if len(prices) < self._lookback:
                continue

            recent_prices = prices[-self._window:]
            mean_price = sum(recent_prices) / self._window
            momentum_score = (recent_prices[-1] - mean_price) / mean_price
            momentum_scores[symbol] = momentum_score

        top_symbols = sorted(momentum_scores.keys(), key=lambda s: momentum_scores[s], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest