"""Category ``future_dependent_ordering``, variant ``buried``.

Defect: ``_peak_levels`` takes an argmax over each name's entire series to locate its highest close,
and the entry rule buys names trading a fixed distance below that peak. The peak is a whole-period
extremum, so on any date before the peak occurs the drawdown the rule measures is a drawdown from a
level the market has not yet reached. Names whose highest price arrives late look permanently cheap
early on, and those are exactly the names that went up.

An extremum over the whole series is the quietest form of this category. It reads as a fact about
the stock rather than as a ranking decision, and the code that consumes it looks like an ordinary
drawdown filter. The rest of the file is a genuine drawdown-and-recovery book with a trend
confirmation, a minimum-history requirement, a per-name cap and a cash buffer.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class DrawdownRecoveryBook(Strategy):
    """Buys names well below their peak that have started to trend back up."""

    rationale = (
        "A large drawdown on its own is not an entry: falling knives keep falling, and buying "
        "depth alone is how a value screen turns into a portfolio of impaired businesses. "
        "Requiring the name to be trading above its own short moving average as well means the "
        "book only takes positions where the decline has at least stopped, which is a weak "
        "confirmation but a cheap one."
    )

    def __init__(
        self,
        levels: pl.DataFrame,
        drawdown: float = 0.25,
        confirm_window: int = 20,
        max_weight: float = 0.10,
        min_history: int = 120,
    ) -> None:
        self._drawdown = drawdown
        self._confirm_window = confirm_window
        self._max_weight = max_weight
        self._min_history = min_history
        self._peaks = _peak_levels(levels)

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=max(self._confirm_window, self._min_history) + 1)
        if closes.height < self._confirm_window:
            return Signal(information_available_at=stamp, weights={})

        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        picks = [
            symbol
            for symbol, price in latest.items()
            if self._deep_enough(symbol, price) and self._confirmed(closes, symbol)
        ]
        return Signal(information_available_at=stamp, weights=self._cap(picks))

    def _deep_enough(self, symbol: str, price: float) -> bool:
        """True when the name trades at least ``drawdown`` below its recorded peak."""
        peak = self._peaks.get(symbol, 0.0)
        if peak <= 0:
            return False
        return price <= peak * (1.0 - self._drawdown)

    def _confirmed(self, closes: pl.DataFrame, symbol: str) -> bool:
        """True when the name is above its own short moving average and has enough history."""
        if symbol not in closes.columns:
            return False
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        if len(values) < self._min_history:
            return False
        recent = values[-self._confirm_window :]
        return values[-1] > sum(recent) / len(recent)

    def _cap(self, picks: list[str]) -> dict[str, float]:
        """Equal weight subject to a per-name cap, with the remainder left in cash."""
        if not picks:
            return {}
        weight = min(1.0 / len(picks), self._max_weight)
        return dict.fromkeys(picks, weight)


def _peak_levels(levels: pl.DataFrame) -> dict[str, float]:
    """Highest close recorded for each name across every session in ``levels``.

    The argmax runs over the complete series, so the value returned for a name whose highest price
    arrives in 2023 is that 2023 price, regardless of which decision date later consumes it.
    """
    ordered = levels.sort(["symbol", "session_date"])
    out: dict[str, float] = {}
    for symbol, group in ordered.group_by("symbol"):
        series = group["adj_close"]
        if series.len() == 0:
            continue
        # THE CHEAT: an extremum over the whole series. Everything downstream measures distance
        # from a level that, for most decision dates, has not been reached yet.
        position = series.arg_max()
        if position is None:
            continue
        key = symbol[0] if isinstance(symbol, tuple) else symbol
        out[str(key)] = float(series[position])
    return out


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
