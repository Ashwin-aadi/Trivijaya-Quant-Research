from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price levels tend to revert to their recent mean. Identifying assets that have "
        "deviated significantly from this mean can provide profitable trading opportunities."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(["symbol", "adj_close"])
        recent_closes = recent_closes.to_pandas()

        mean_close = recent_closes.groupby("symbol").mean().rename(columns={"adj_close": "mean_adj_close"})
        std_close = recent_closes.groupby("symbol").std().rename(columns={"adj_close": "std_adj_close"})

        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if mean_close.loc[mean_close['symbol'] == symbol].empty or \
               std_close.loc[std_close['symbol'] == symbol].empty:
                continue

            recent_mean = mean_close.loc[mean_close['symbol'] == symbol]['mean_adj_close'].values[0]
            recent_std = std_close.loc[std_close['symbol'] == symbol]['std_adj_close'].values[0]

            latest_close = view.latest_close()[symbol]
            score = (latest_close - recent_mean) / recent_std

            reversion_scores[symbol] = score

        top_symbols = sorted(reversion_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest