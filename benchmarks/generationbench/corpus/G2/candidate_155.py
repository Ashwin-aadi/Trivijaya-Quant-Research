from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the momentum of a stock "
        "and its volatility. The idea is that stocks with strong recent performance and high "
        "volatility may be overreacting to short-term noise, presenting an opportunity for mean reversion."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 15) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._volatility_window)

        if closes.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        volatility_scores: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]

            if len(values) < self._momentum_window + self._volatility_window:
                continue

            momentum_score = (values[-1] - values[0]) / max(values)
            volatility_score = pl.col(symbol).std().item()
            momentum_scores.append(momentum_score)
            volatility_scores.append(volatility_score)

        combined_scores = [m * v for m, v in zip(momentum_scores, volatility_scores)]
        top_symbols = sorted(zip(view.symbols, combined_scores), key=lambda x: -x[1])[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest