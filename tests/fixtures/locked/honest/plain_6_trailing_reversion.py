"""A plain trailing z-score mean-reversion rule: buy names far below their recent average.

Negative control: the mean and standard deviation are both computed over ``view.closes(lookback)``,
already bounded to sessions strictly before the decision date. The most recent visible close is
compared against a trailing statistic that includes it, exactly as a Bollinger-style band would --
this is standard practice, not a look outside the window.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingMeanReversionStrategy(Strategy):
    """Buy names whose latest close sits several trailing standard deviations below the mean."""

    rationale = (
        "A price that has drifted several trailing standard deviations below its own recent "
        "average has, historically, shown a mild tendency to snap back toward that average over "
        "short horizons. This strategy buys the most stretched names in the visible universe as "
        "a plain reversion bet, made with no claim it clears trading costs."
    )

    def __init__(self, lookback: int = 20, z_entry: float = 1.0, top_n: int = 5) -> None:
        self.lookback = lookback
        self.z_entry = z_entry
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        recent = view.history(1)
        decision_date = date.min if recent.is_empty() else recent["session_date"].max()
        assert isinstance(decision_date, date)
        closes = view.closes(self.lookback)
        if closes.height < self.lookback:
            return Signal(information_available_at=decision_date, weights={})
        last_row = closes.tail(1).row(0, named=True)
        z_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            series = closes[symbol]
            if series.null_count() > 0:
                continue
            trailing_mean = series.mean()
            trailing_std = series.std()
            current = last_row.get(symbol)
            if current is None or trailing_mean is None or not trailing_std:
                continue
            z_scores[symbol] = (current - trailing_mean) / trailing_std
        candidates = sorted(
            (item for item in z_scores.items() if item[1] < -self.z_entry), key=lambda kv: kv[1]
        )[: self.top_n]
        weights: dict[str, float] = {}
        if candidates:
            share = 1.0 / len(candidates)
            for symbol, _z in candidates:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
