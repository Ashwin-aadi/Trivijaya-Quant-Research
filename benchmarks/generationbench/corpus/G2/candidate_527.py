from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "The strategy leverages two weakly related characteristics: the 20-day momentum of "
        "the stock and its implied volatility. Momentum can indicate future price movement "
        "trends, while implied volatility suggests potential for large moves in prices. By "
        "combining these signals, we aim to identify stocks that are both trending positively "
        "and have high implied volatility, suggesting they could experience significant price "
        "action."
    )

    def __init__(self, momentum_window: int = 20, top_n: int = 5) -> None:
        self._momentum_window = momentum_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window)
        if closes.height < self._momentum_window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._momentum_window + 1:
                continue
            momentum_score = (values[-1] - values[0]) / abs(values[0])
            if momentum_score > 0.05:  # Simple threshold for positive momentum
                momentum_picks.append(symbol)

        picks = _filter_high_volatility(momentum_picks, view)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _filter_high_volatility(picks: list[str], view: MarketView) -> list[str]:
    closes = view.closes(lookback=20)
    high_volatility_symbols: list[str] = []
    for symbol in picks:
        if symbol not in closes.columns:
            continue
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(values) < 20 + 1:
            continue
        implied_volatility = _implied_volatility(values)
        if implied_volatility > 0.2:  # Simple threshold for high volatility
            high_volatility_symbols.append(symbol)

    return high_volatility_symbols


def _implied_volatility(prices: list[float]) -> float:
    prices.sort(descending=True)
    log_returns = [(prices[i] - prices[i + 1]) / abs(prices[i + 1]) for i in range(len(prices) - 1)]
    mean_log_return = sum(log_returns) / len(log_returns)
    variance = sum([x ** 2 for x in log_returns]) / len(log_returns)
    volatility = (mean_log_return + 0.5 * variance) * 252
    return volatility