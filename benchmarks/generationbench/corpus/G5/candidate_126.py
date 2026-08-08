from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "A stock with a higher recent relative strength compared to the NIFTY 100 index "
        "is likely to outperform in the near term. This strategy aims to identify such stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty_100 = [symbol for symbol in view.symbols if "NIFTY" in symbol]
        if not nifty_100:
            return Signal(information_available_at=stamp, weights={})

        nifty_100_close = view.closes(lookback=self._window).select(
            pl.col(nifty_100).mean().alias("nifty_mean")
        )

        relative_strengths: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if "NIFTY" not in symbol or symbol == nifty_100[0]:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(close_series[-self._window:]) / self._window
            relative_strength = mean_close / float(nifty_100_close.select("nifty_mean")[symbol])
            if relative_strength > 1.2:
                relative_strengths.append((symbol, relative_strength))

        if not relative_strengths:
            return Signal(information_available_at=stamp, weights={})

        top_n_stocks = sorted(relative_strengths, key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_n_stocks)
        return Signal(
            information_available_at=stamp, weights=dict(top_n_stocks)
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest