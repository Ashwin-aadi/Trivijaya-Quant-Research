from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumAndVolatility(Strategy):
    rationale = (
        "This strategy aims to exploit the idea that securities with strong recent momentum "
        "and low historical volatility are likely to outperform. The combination of positive "
        "momentum and stable performance tends to indicate continued strength in demand."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 60) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            # Calculate daily returns
            rets = [float(v) / float(history[symbol].shift(1).item()) - 1.0 for v in history[symbol][1:].to_list()]
            # Momentum score: average of recent returns
            momentum_scores[symbol] = sum(rets[-self._momentum_window:]) / self._momentum_window

            # Volatility score: standard deviation of recent prices
            vol = pl.col(symbol).std().item()
            volatility_scores[symbol] = 1.0 / (1 + vol)

        combined_score = {symbol: momentum_scores[symbol] * volatility_scores[symbol] for symbol in momentum_scores}
        sorted_symbols = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)

        picks = [s[0] for s in sorted_symbols[:5]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest