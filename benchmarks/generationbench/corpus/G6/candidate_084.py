from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Utilizing a 120-day rolling return to measure momentum provides more stable signals "
        "for long-term equity strategy. This approach ensures that the portfolio is constructed "
        "based on sustained performance metrics, balancing between high recent performance and "
        "risk management."
    )

    def __init__(self, window: int = 120, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            rolling_return = (values[-1] / values[0]) - 1.0
            momentum_scores[symbol] = rolling_return

        sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [stock for stock, _ in sorted_scores[: self._top_n]]

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight_per_stock = 1.0 / len(top_stocks)
        stop_loss = -0.05
        weights = {s: weight_per_stock for s in top_stocks}

        # Ensure no single stock exceeds the stop-loss threshold
        for stock, w in list(weights.items()):
            if momentum_scores[stock] < stop_loss:
                del weights[stock]

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest