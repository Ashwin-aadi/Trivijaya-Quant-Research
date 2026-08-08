from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy captures trends by scaling trading positions based on recent market "
        "volatility. Higher volatility leads to smaller position sizes to mitigate risk during "
        "turbulent periods."
    )

    def __init__(self, sma_window: int = 50, vol_threshold_long: float = 0.02,
                 vol_threshold_short: float = 0.04, max_stocks: int = 25) -> None:
        self._sma_window = sma_window
        self._vol_threshold_long = vol_threshold_long
        self._vol_threshold_short = vol_threshold_short
        self._max_stocks = max_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=20 + self._sma_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("close") / pl.col("close").shift(self._sma_window) - 1).alias("return_ratio"),
                pl.col("close").mean().alias("sma"),
                (pl.col("adj_close").rolling_std(window=self._sma_window, center=True) / pl.col("sma")).alias("volatility")
            )
        )

        if sma.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sma.columns:
                continue
            return_ratio = float(sma[symbol]["return_ratio"].last())
            volatility = float(sma[symbol]["volatility"].last())
            sma_value = float(sma[symbol]["sma"].last())

            if (return_ratio > 0 and volatility <= self._vol_threshold_long and
                    history[history["symbol"] == symbol]["close"].max() > sma_value):
                picks.append(symbol)

            elif (return_ratio < 0 and volatility >= self._vol_threshold_short and
                  history[history["symbol"] == symbol]["close"].min() < sma_value):
                picks.append(symbol)

        if len(picks) > self._max_stocks:
            picks = sorted(picks, key=lambda x: _latest_close(view, x), reverse=True)[:self._max_stocks]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def _latest_close(view: MarketView, symbol: str) -> float:
    closes = view.closes().select([pl.col(symbol)])
    if not closes.height:
        return 0.0
    return float(closes[symbol].last())