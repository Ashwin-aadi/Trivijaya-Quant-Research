from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers in the past are likely to continue "
        "outperforming them in the future. This is based on the assumption of mean reversion and "
        "momentum effects within the market."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_period)
        if closes.height < self._lookback_period or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_dict(as_series=False)
        market_avg = sum(mean_close.values()) / len(view.symbols)

        strength_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_close:
                continue
            score = (mean_close[symbol] - market_avg) / market_avg
            strength_scores[symbol] = score

        sorted_strengths = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_strengths[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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