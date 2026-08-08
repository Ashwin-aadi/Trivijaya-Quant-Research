from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and accessibility. Highly liquid stocks "
        "tend to have tighter bid-ask spreads and are less prone to price distortions due to "
        "insufficient trading volume. By equal-weighting these more accessible stocks, the "
        "strategy aims to benefit from reduced transaction costs and better execution of trades."
    )

    def __init__(self, liquidity_threshold: float = 10_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=20)
        if closes.height < 20 or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        liquidity = {}
        for symbol in view.symbols:
            recent_closes = [float(v) for v in closes[symbol].to_list()[-20:]]
            median_close = float(pl.Series(recent_closes).median())
            daily_volumes = (
                view.history(lookback=20)
                .filter(pl.col("symbol") == symbol)
                .select(pl.col("volume"))
                .to_numpy()[0]
            )
            if len(daily_volumes) >= 10:  # At least half of the days should have volume data
                average_volume = sum(daily_volumes) / 20
                liquidity[symbol] = median_close * average_volume

        sorted_liquidity = dict(sorted(liquidity.items(), key=lambda item: -item[1]))
        picks = [k for k, v in sorted_liquidity.items() if v >= self._threshold]
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