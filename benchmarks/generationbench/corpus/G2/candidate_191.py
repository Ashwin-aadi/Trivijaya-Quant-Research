from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines a momentum signal with a volatility filter. "
        "Momentum suggests that stocks that have performed well recently are likely to continue performing well. "
        "Volatility filters out highly volatile stocks, as they may be riskier and less suitable for the current market environment."
    )

    def __init__(self, momentum_window: int = 20, volatility_threshold: float = 1.5) -> None:
        self._momentum_window = momentum_window
        self._volatility_threshold = volatility_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + 1)
        if closes.height < self._momentum_window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].to_list()]
            latest_close = values[-1]
            if len(values) < self._momentum_window + 1 or any(pl.col("adj_close").is_nan()):
                continue
            momentum_score = (latest_close - min(values[:-1])) / max(values[:-1])
            momentum_scores[symbol] = momentum_score

        # Filter by volatility
        history = view.history(lookback=self._momentum_window)
        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].select(["adj_close", "volume"]).to_pandas().dropna().apply(lambda x: x["adj_close"] - latest_close, axis=1).to_list()]
            volatility = (sum([abs(value) for value in values]) / len(values)) ** 0.5
            volatilities[symbol] = volatility

        filtered_symbols = [symbol for symbol in momentum_scores if symbol in volatilities and volatilities[symbol] < self._volatility_threshold]
        scores = {symbol: (momentum_scores[symbol] * 2 - volatilities[symbol]) for symbol in filtered_symbols}

        top_n_symbols = sorted(scores, key=scores.get, reverse=True)[:5]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest