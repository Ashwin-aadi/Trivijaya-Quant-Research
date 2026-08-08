from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolatility(Strategy):
    rationale = (
        "This strategy combines momentum and volatility to identify stocks with strong"
        " recent performance but low future expected volatility. The idea is that such"
        " stocks are potentially undervalued and have a higher chance of outperforming."
    )

    def __init__(self, momentum_window: int = 15, volatility_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].to_list()]
            if len(close_values) < self._momentum_window + self._volatility_window:
                continue

            # Calculate momentum score (log return over the last 20 days)
            log_return = (
                pl.col("adj_close").shift(-self._momentum_window).log()
                - pl.col("adj_close").log()
            ).mean()
            momentum_scores[symbol] = float(log_return)

            # Calculate volatility score (rolling standard deviation of returns over the last 10 days)
            returns = [
                float((close_values[i + 1] / close_values[i]) - 1.0) for i in range(len(close_values) - 1)
            ]
            if len(returns) < self._volatility_window:
                continue
            volatility_score = pl.Series(returns).rolling_std(window=self._volatility_window)
            volatility_scores[symbol] = float(volatility_score[-1])

        # Combine scores using a simple weighted average (momentum_weight * momentum + volatility_weight * volatility)
        combined_scores = {
            symbol: 0.7 * momentum_scores[symbol] - 0.3 * volatility_scores[symbol]
            for symbol in view.symbols
        }

        top_symbols = sorted(combined_scores, key=combined_scores.get, reverse=True)[:5]

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