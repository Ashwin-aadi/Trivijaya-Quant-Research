from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityCompression(Strategy):
    rationale = (
        "This strategy exploits periods of high or low volatility by identifying stocks "
        "with significant dispersion in their intraday price movements. By entering trades "
        "based on high implied volatility rank (IVR) and exiting when IVI falls below a threshold, "
        "we aim to capture profitable trading opportunities."
    )

    def __init__(self, window: int = 30, top_n: int = 10, ivr_threshold_high: float = 2.0, ivr_threshold_low: float = 0.5) -> None:
        self._window = window
        self._top_n = top_n
        self._ivr_threshold_high = ivr_threshold_high
        self._ivr_threshold_low = ivr_threshold_low

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_ivr_stocks = []
        for symbol in view.symbols:
            hist_data = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_prices = [float(v) for v in hist_data["open"].to_list()]
            close_price = float(view.latest_close()[symbol])

            if len(open_prices) < self._window:
                continue

            ivi = (max(open_prices) - min(open_prices)) / close_price
            if ivi >= self._ivr_threshold_high and pl.col("volume").mean().over(["symbol"]).filter(pl.col("session_date") == hist_data["session_date"].max())[0] > 1.5 * pl.col("volume").mean().over(["symbol"]):
                high_ivr_stocks.append(symbol)

        if not high_ivr_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_ivr_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_ivr_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest