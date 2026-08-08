from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the fact that stocks with strong relative performance "
        "in recent periods tend to continue outperforming in the future. This effect is driven by "
        "the tendency of investors to chase past winners."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback or history.width < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol)
            ).sort("session_date")
            if symbol_history.height < self._lookback + 1:
                continue

            recent_closes = [float(v) for v in symbol_history["close"].to_list()[-self._lookback:]]
            mean_close = sum(recent_closes) / len(recent_closes)
            momentum_score = (recent_closes[-1] - mean_close) / mean_close
            momentum_scores[symbol] = momentum_score

        sorted_symbols = [
            symbol for symbol, _ in sorted(momentum_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        top_5_symbols = sorted_symbols[:5]
        if not top_5_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_5_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_5_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest