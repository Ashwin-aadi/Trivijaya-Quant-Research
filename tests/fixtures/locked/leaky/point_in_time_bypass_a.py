"""Opening-range breakout on NIFTY constituents.

Buys names whose session open clears their recent trading range by a meaningful margin, on the
view that an early gap higher signals fresh institutional demand that tends to persist through
the session. Position size is scaled down for names with a wide recent range, since breaking out
of a choppy range is far less informative than breaking out of a tight one.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class OpeningRangeBreakout(Strategy):
    """Buys symbols whose session open breaks meaningfully above their recent trading range."""

    rationale = (
        "A session open that clears the prior fortnight's trading range often marks the start "
        "of a fresh institutional accumulation phase, since large orders are usually worked in "
        "over several sessions rather than placed all at once. Sizing down names with a wide "
        "recent range keeps the strategy from over-weighting breakouts that are really just "
        "noise."
    )

    def __init__(self, full_panel: pl.DataFrame, range_window: int = 10, top_n: int = 8) -> None:
        # Kept alongside the strategy object so each session's own opening print is on hand the
        # moment it prints, rather than waiting on the next daily view refresh.
        self._panel = full_panel
        self._range_window = range_window
        self._top_n = top_n

    def _opening_prices(self, as_of: date, symbols: tuple[str, ...]) -> dict[str, float]:
        row = self._panel.filter(
            (pl.col("session_date") == as_of) & (pl.col("symbol").is_in(list(symbols)))
        )
        names = row["symbol"].to_list()
        prices = row["open"].to_list()
        return dict(zip(names, prices, strict=True))

    def generate(self, view: MarketView) -> Signal:
        history = view.history(self._range_window)
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        ranges = history.group_by("symbol").agg(
            [pl.col("adj_close").max().alias("hi"), pl.col("adj_close").min().alias("lo")]
        )
        opens = self._opening_prices(view.as_of, view.symbols)
        names = ranges["symbol"].to_list()
        his = ranges["hi"].to_list()
        los = ranges["lo"].to_list()
        scored: list[tuple[str, float, float]] = []
        for symbol, hi, lo in zip(names, his, los, strict=True):
            open_price = opens.get(symbol)
            if open_price is None or hi <= lo:
                continue
            band = hi - lo
            if open_price > hi:
                scored.append((symbol, (open_price - hi) / band, band))
        scored.sort(key=lambda row: row[1], reverse=True)
        chosen = scored[: self._top_n]
        weights: dict[str, float] = {}
        for symbol, _, band in chosen:
            weights[symbol] = 1.0 / band if band > 0 else 0.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return Signal(information_available_at=view.as_of, weights=weights)
