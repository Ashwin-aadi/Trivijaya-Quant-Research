from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to identify and capitalize on trends by normalizing "
        "price movements using historical volatility. High volatility periods are "
        "expected to smooth out price movements, making it easier to detect true "
        "trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].unique()) <= 1:
                continue

            # Calculate daily returns
            close_values = [float(v) for v in history[symbol].to_list()]
            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] if close_values[i-1] != 0 else float('nan') for i in range(1, len(close_values))]

            # Calculate mean return and standard deviation
            mean_return = sum(returns) / max(len(returns), 1)
            std_dev = (sum([(r - mean_return)**2 for r in returns if not pl.col("r").is_nan()]) / max(len(returns) - 1, 1))**0.5

            # Normalize the last close by its historical volatility
            normalized_close = history[symbol][-1] / std_dev if std_dev != 0 else float('nan')

            trends[symbol] = normalized_close

        top_trends = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        if not top_trends:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_trends)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_trends}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest