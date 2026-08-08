from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the small-firm effect by focusing on stocks with high "
        "trading volumes, ensuring diversification and mitigating concentration risk. "
        "By equally weighting selected stocks each trading day, we aim to capture higher "
        "returns associated with market inefficiencies in smaller firms."
    )

    def __init__(self, min_volume: float = 1_000_000, top_n: int = 50) -> None:
        self._min_volume = min_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=1)
        if closes.height < 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            daily_volume = float(closes[symbol].to_list()[0])
            if daily_volume >= self._min_volume:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest