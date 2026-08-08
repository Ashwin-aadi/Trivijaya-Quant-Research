from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies stocks with strong relative strength over a "
        "recent period. These stocks are expected to continue outperforming in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbol_column = pl.Series("symbol", latest_closes.columns[1:], pl.Utf8)

        momentum_scores: dict[str, float] = {}
        for _, row in history.iter_rows():
            symbol = str(row["symbol"])
            scores = [float(v) for v in latest_closes[symbol].drop_nulls().to_list()]
            if len(scores) < self._window:
                continue
            recent_close = max(scores)
            rank = (pl.col("adj_close") / pl.col("adj_close").shift(1).fill_null(pl.lit(1.0)) - 1.0).rank(method="dense", descending=True)
            momentum_scores[symbol] = float(rank.head(self._window)[0])

        top_symbols = sorted(momentum_scores, key=momentum_scores.get, reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest