from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the observation that assets with higher "
        "volatility are more likely to continue trending in their recent direction. By scaling "
        "trades by volatility, we can increase our exposure during periods of high volatility and "
        "reduce it during low volatility, aiming for both capturing trends and reducing risk."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._window).select(
                pl.col("symbol"), pl.col("session_date"), pl.col("adj_close")
            )
            returns = (history.select(pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0)
                       .drop_nulls().sort("session_date"))
            if returns.height < self._window:
                continue
            rolling_mean = returns.sort("session_date").select(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean()
            )
            recent_return = returns.sort("session_date", descending=True).head(5)["adj_close"].last() - \
                            returns.sort("session_date").head(5)["adj_close"].last()

            if rolling_mean.is_empty():
                continue
            volatility = (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).std().over(pl.arange(1, self._window + 1)).mean()
            weight = recent_return * max(volatility, 0.01)
            volatility_scaled_weights[symbol] = float(weight)

        total_weight = sum([abs(v) for v in volatility_scaled_weights.values()])
        if total_weight == 0:
            return Signal(information_available_at=stamp, weights={})

        scaled_weights = {s: w / total_weight for s, w in volatility_scaled_weights.items()}
        return Signal(
            information_available_at=stamp,
            weights=scaled_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest