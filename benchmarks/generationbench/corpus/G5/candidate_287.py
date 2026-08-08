from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolatility(Strategy):
    rationale = (
        "This strategy combines two signals: a momentum signal based on 50-day returns and "
        "a volatility signal based on 20-day standard deviation. Both are meant to capture "
        "momentum and stability in the market, which can help identify strong but stable "
        "stocks for investment."
    )

    def __init__(self, momentum_window: int = 50, vol_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._vol_window))
        if history.height < max(self._momentum_window, self._vol_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._momentum_window + 1:
                continue

            # Calculate momentum score
            momentum_score = (prices[-1] - prices[0]) / prices[0]
            momentum_scores[symbol] = momentum_score

            # Calculate volatility over the last 20 days
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]
            volatilities[symbol] = pl.DataFrame({"returns": returns}).select(
                (pl.col("returns") ** 2).mean().alias("variance")
            ).tail(self._vol_window)[0]["variance"] ** 0.5

        # Filter symbols based on momentum and volatility
        candidates = sorted(momentum_scores, key=lambda x: -momentum_scores[x])
        filtered_symbols = [s for s in candidates if volatilities[s] < pl.Series(volatilities).mean()]

        top_3_symbols = filtered_symbols[:3]
        if not top_3_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_3_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_3_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest