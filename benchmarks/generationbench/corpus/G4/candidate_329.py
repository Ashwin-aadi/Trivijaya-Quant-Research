from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategy identifies key support and resistance levels. "
        "Once a breakout is confirmed, it looks for further momentum to generate trade signals."
    )

    def __init__(self, lookback: int = 50, margin: float = 0.02) -> None:
        self._lookback = lookback
        self._margin = margin

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        candidates: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique():
                continue
            data = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            opens = [float(v) for v in data["open"].to_list()]
            closes = [float(v) for v in data["close"].to_list()]

            # Identify support and resistance levels from the lookback period
            supports = pl.DataFrame({"date": data["session_date"], "level": data["low"]}).sort(by="level").head(1)
            resistances = pl.DataFrame({"date": data["session_date"], "level": data["high"]}).sort(by="level", descending=True).head(1)

            if supports.height == 0 or resistances.height == 0:
                continue

            support_level, resistance_level = float(supports.select("level")), float(resistances.select("level"))
            breakout_high = max(opens[0], closes[0]) > (resistance_level * (1 + self._margin))
            breakout_low = min(opens[-1], closes[-1]) < (support_level * (1 - self._margin))

            if breakout_high:
                price_threshold = resistance_level * (1 + self._margin)
                if max(closes) >= price_threshold and data.select(pl.col("close") > price_threshold).height > 0:
                    candidates[symbol] = price_threshold
            elif breakout_low:
                price_threshold = support_level * (1 - self._margin)
                if min(closes) <= price_threshold and data.select(pl.col("close") < price_threshold).height > 0:
                    candidates[symbol] = price_threshold

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        # Rank candidates by strength
        ranked_candidates = sorted(candidates.items(), key=lambda x: (x[1], float(view.closes().select(x[0]).to_list()[-1])), reverse=True)
        top_symbol, _ = ranked_candidates[0]

        weight = 1.0 / len(ranked_candidates) if ranked_candidates else 0
        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight} if ranked_candidates else {},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().item()
    assert isinstance(newest, date)
    return newest