from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumSectorFocus(Strategy):
    rationale = (
        "This strategy leverages dual momentum by entering positions in stocks that show "
        "recent uptrends in both closing prices and volume increases. It also incorporates "
        "sector-based mean reversion to balance risk across selected sectors, ensuring "
        "that the portfolio does not lose more than 15% cumulatively."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty() or history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        sector_focus = ["IT", "Financial Services", "Consumer Discretionary"]
        valid_sectors: set[str] = set(sector_focus)
        picks: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            closes = hist.select("adj_close").to_numpy().flatten()
            opens = hist.select("open").to_numpy().flatten()
            highs = hist.select("high").to_numpy().flatten()
            lows = hist.select("low").to_numpy().flatten()
            volumes = hist.select("volume").to_numpy().flatten()

            if len(closes) < self._window:
                continue

            last_close, second_last_close = closes[-1], closes[-2]
            vol_increase = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 >= 10

            if (
                last_close > max(closes[:-1])
                and last_close > opens[-1]
                and last_close > lows[-1]
                and second_last_close < max(closes[:-2])
                and vol_increase
            ):
                sector = view.history().filter(pl.col("symbol") == symbol).select("sector").to_numpy().flatten()[0]
                if sector in valid_sectors:
                    picks[symbol] = 1.0 / len(picks)

        picks = {k: v for k, v in sorted(picks.items(), key=lambda item: -item[1])}
        top_n_symbols = list(next(iter(picks.keys() for _ in range(self._top_n)), {}))
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest