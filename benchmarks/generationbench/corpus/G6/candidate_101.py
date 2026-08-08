from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DonchianBreakout(Strategy):
    rationale = (
        "Utilizing Donchian Channels to identify breakouts ensures stocks surpass significant "
        "resistance or support levels with strong volume. This provides a clear and reliable method "
        "for spotting potential trends."
    )

    def __init__(self, lookback: int = 20, threshold: float = 1.05) -> None:
        self._lookback = lookback
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < 2 * self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        filtered_history = (
            history.filter(pl.col("symbol").is_in(symbols))
                   .sort("session_date")
                   .group_by("symbol")
                   .agg([
                       (pl.col("high").tail(self._lookback).max().alias("donchian_high")),
                       (pl.col("low").head(self._lookback).min().alias("donchian_low"))
                   ])
        )

        breakout_signals = []
        for symbol in symbols:
            hist = filtered_history.filter(pl.col("symbol") == symbol)
            if not hist.height or hist["donchian_high"].item() <= hist["donchian_low"].item():
                continue

            last_close = view.latest_close()[symbol]
            open_price = float(view.history(lookback=1)["open"][0][symbol])
            volume = float(view.history(lookback=1)["volume"][0][symbol])

            if open_price > hist["donchian_high"].item() * self._threshold and volume > 50:
                breakout_signals.append(symbol)
            elif open_price < hist["donchian_low"].item() / self._threshold and volume > 50:
                breakout_signals.append(symbol)

        weights = {symbol: 1.0 / len(breakout_signals) for symbol in breakout_signals}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest