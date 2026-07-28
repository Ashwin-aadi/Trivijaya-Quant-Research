"""A plain inverse-volatility tilt: quieter names get more weight, a simple risk-parity idea.

Negative control: daily returns are built with a single, positive ``shift(1)`` -- a trailing lag
that compares each session to the one immediately before it, never to a later one -- applied to a
column already bounded to the trailing volatility window.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class InverseVolatilityTilt(Strategy):
    """Weight names inversely to their trailing return volatility."""

    rationale = (
        "Quieter names contribute less unrewarded risk per unit of capital, so weighting a "
        "basket inversely to trailing volatility is a standard risk-parity construction rather "
        "than a return forecast. It says nothing about which names will go up, only how much of "
        "each to hold if all of them are held."
    )

    def __init__(self, vol_window: int = 60, max_names: int = 8) -> None:
        self.vol_window = vol_window
        self.max_names = max_names

    @staticmethod
    def _stamp(view: MarketView) -> date:
        recent = view.history(1)
        if recent.is_empty():
            return date.min
        latest = recent["session_date"].max()
        assert isinstance(latest, date)
        return latest

    def generate(self, view: MarketView) -> Signal:
        decision_date = self._stamp(view)
        closes = view.closes(self.vol_window + 1)
        if closes.height < self.vol_window + 1:
            return Signal(information_available_at=decision_date, weights={})
        inv_vol: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price = closes[symbol]
            if price.null_count() > 0:
                continue
            # A positive shift(1) is a trailing lag: row i is compared with row i-1, the session
            # immediately before it. This is the ordinary definition of a daily return.
            daily_return = (price / price.shift(1) - 1.0).drop_nulls()
            vol = daily_return.std()
            if vol is None or vol <= 0:
                continue
            inv_vol[symbol] = 1.0 / vol
        ranked = sorted(inv_vol.items(), key=lambda kv: kv[1], reverse=True)[: self.max_names]
        selected = dict(ranked)
        total = sum(selected.values())
        weights: dict[str, float] = {}
        if total > 0:
            for symbol, score in selected.items():
                weights[symbol] = score / total
        return Signal(information_available_at=decision_date, weights=weights)
