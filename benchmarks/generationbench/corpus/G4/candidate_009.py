from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the low-volatility anomaly by tilting towards stocks with lower "
        "historical volatility. The empirical evidence suggests that these stocks provide higher "
        "risk-adjusted returns over time."
    )

    def __init__(self, window: int = 252, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            returns = [(prices[i] / prices[i - 1] - 1.0) for i in range(1, len(prices))]
            volatility_score = pl.Series(returns).std()
            volatility_scores[symbol] = float(volatility_score)

        ranked_symbols = sorted(volatility_scores.items(), key=lambda item: item[1])
        selected_symbols = [symbol for symbol, _ in ranked_symbols[: self._top_n]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest