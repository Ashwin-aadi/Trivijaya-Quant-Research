from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following is a strategy that seeks to capture trends while "
        "limiting drawdowns by adjusting position sizes based on volatility. High-volatility "
        "periods suggest greater market uncertainty and thus smaller positions, while low-"
        "volatility periods may indicate stable markets and larger positions."
    )

    def __init__(self, window: int = 20, risk_factor: float = 1.0) -> None:
        self._window = window
        self._risk_factor = risk_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = (
            history
            .sort("session_date")
            .select([
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            ])
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("mean_return"), 
                 pl.col("return").std().alias("volatility"))
        )

        volatility_scaled_weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in symbol_returns.columns:
                continue
            mean_return = float(symbol_returns.filter(pl.col("symbol") == symbol)["mean_return"].item())
            volatility = float(symbol_returns.filter(pl.col("symbol") == symbol)["volatility"].item())

            if volatility > 0.0:  # Avoid division by zero
                weight = self._risk_factor * (mean_return / volatility)
            else:
                weight = 0.0

            volatility_scaled_weights[symbol] = weight

        total_weight_sum = sum(volatility_scaled_weights.values())
        weights = {symbol: weight / total_weight_sum for symbol, weight in volatility_scaled_weights.items() if weight > 0}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest