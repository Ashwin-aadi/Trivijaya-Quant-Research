from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BollingerBandsStrategy(Strategy):
    rationale = (
        "This strategy exploits dispersion or range compression in Indian equity markets by "
        "identifying stocks where the price action is outside Bollinger Bands (indicating increased"
        "volatility) or where bands have narrowed significantly (indicating reduced volatility)."
    )

    def __init__(self, window: int = 20, multiplier: float = 2.0, top_n: int = 30) -> None:
        self._window = window
        self._multiplier = multiplier
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the central band (20-day moving average of close prices)
        closes = history.select(
            pl.col("symbol"),
            pl.col("close").rolling_mean(window_size=self._window).alias("central_band"),
        )
        
        # Compute returns and Bollinger Bands
        std_dev = (
            history.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("returns")
            )
            .select(
                pl.col("symbol"),
                pl.col("returns").rolling_std(window_size=self._window, center=True).alias("std_dev")
            )
        )

        bollinger_bands = (
            std_dev.join(closes, on="symbol", how="left")
            .with_columns(
                (pl.col("central_band") + self._multiplier * pl.col("std_dev")).alias("upper_band"),
                (pl.col("central_band") - self._multiplier * pl.col("std_dev")).alias("lower_band"),
            )
        )

        # Identify stocks for long and short positions
        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in bollinger_bands.columns or symbol not in history.columns:
                continue

            recent_closes = [float(v) for v in history[symbol].to_list()[-self._window:]]
            current_close = history[history["symbol"] == symbol]["close"].last()
            recent_upper_band = bollinger_bands[(bollinger_bands["symbol"] == symbol)]["upper_band"]
            recent_lower_band = bollinger_bands[(bollinger_bands["symbol"] == symbol)]["lower_band"]

            if current_close > recent_upper_band.item():
                signals[symbol] = -0.1
            elif current_close < recent_lower_band.item():
                signals[symbol] = 0.1

        # Rank and select top N stocks for long or short positions
        ranked_signals = {k: v for k, v in sorted(signals.items(), key=lambda item: abs(item[1]), reverse=True)}
        selected_stocks = list(ranked_signals.keys())[:self._top_n]

        if not selected_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest