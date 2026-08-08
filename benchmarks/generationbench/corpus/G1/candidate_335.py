from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy leverages a combination of recent momentum and short-term volatility to identify "
        "potential trading opportunities. High momentum signals are often followed by price reversals or continuation, "
        "while low volatility can indicate strong support or demand."
    )

    def __init__(self, window1: int = 20, window2: int = 5) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window1)

        if closes.height < self._window1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            score = (values[-1] - min(values)) / (max(values) - min(values))
            momentum_scores.append(score)

        volatility_scores: list[float] = []
        history = view.history(lookback=self._window2)
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = (
                pl.col("close").shift(-1) / pl.col("close") - 1.0
            ).drop_nulls().to_list()
            volatility = ((abs(daily_returns)) ** 2).mean() ** 0.5
            volatility_scores.append(volatility)

        if not momentum_scores or not volatility_scores:
            return Signal(information_available_at=stamp, weights={})

        combined_scores = [
            (momentum_scores[i] * 0.6 + volatility_scores[i] * 0.4) for i in range(len(view.symbols))
        ]
        top_symbols = [symbol for _, symbol in sorted(zip(combined_scores, view.symbols), reverse=True)[:3]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest