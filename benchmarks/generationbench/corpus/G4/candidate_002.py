from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the tendency for assets to exhibit mean-reverting behavior after large price movements (volatility). "
        "During periods of high volatility, markets often experience extreme price swings, and following these trends can yield profits."
    )

    def __init__(self, window: int = 20, upper_vol_threshold: float = 30.0, lower_vol_threshold: float = 10.0) -> None:
        self._window = window
        self._upper_vol_threshold = upper_vol_threshold
        self._lower_vol_threshold = lower_vol_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol) for symbol in view.symbols]
        realized_volatility = self._calculate_realized_volatility(history)

        signals: dict[str, float] = {}
        for symbol in symbols:
            vol_rank = 1 if realized_volatility[symbol] > self._upper_vol_threshold else -1
            weight = 0.2 / (abs(vol_rank) + 1)
            if vol_rank == 1 and symbol not in signals:
                signals[symbol] = weight
            elif vol_rank == -1 and symbol in signals:
                del signals[symbol]

        return Signal(information_available_at=stamp, weights={symbol: weight for symbol, weight in signals.items()})

    def _calculate_realized_volatility(self, history: pl.DataFrame) -> dict[str, float]:
        realized_vols = {}
        symbols = [str(symbol) for symbol in view.symbols]
        for symbol in symbols:
            df = history.select(pl.col("symbol").eq(symbol).alias("is_symbol"),
                               "open", "high", "low", "close")
            log_returns = (df["close"] / df["close"].shift(1) - 1.0).drop_nulls().to_list()
            realized_volatility = ((sum(log_returns) ** 2 / len(log_returns)) * self._window).sqrt() * 100
            realized_vols[symbol] = float(realized_volatility)
        return realized_vols


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest