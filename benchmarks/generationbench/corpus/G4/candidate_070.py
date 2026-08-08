from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "The strategy exploits historical seasonal trends in the Indian equity market to "
        "identify and trade stocks during favorable periods. By analyzing past performance "
        "and current momentum indicators, we can predict potential gains from upcoming seasonality effects."
    )

    def __init__(self, lookback_period: int = 5, top_n: int = 20) -> None:
        self._lookback_period = lookback_period
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        monthly_returns: dict[str, float] = {}
        volume_weighted_prices: dict[str, float] = {}

        for symbol in symbols:
            monthly_history = (
                history.filter(pl.col("session_date").dt.month().cast(pl.Int32))
                .sort("session_date")
                .group_by(pl.col("session_date").dt.month())
                .agg(
                    [
                        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias(
                            "monthly_return"
                        ),
                        (pl.col("adj_close") * pl.col("volume")).sum()
                        / pl.col("volume").sum().alias("volume_weighted_price"),
                    ]
                )
            )

            if not monthly_history.is_empty():
                avg_monthly_return = (
                    monthly_history.select(pl.col("monthly_return").mean()).item()
                )
                volume_weighted_price = (
                    monthly_history.select(
                        (pl.col("volume_weighted_price") / pl.lit(1)).alias(
                            "volume_weighted_price"
                        )
                    ).item()
                )

                if avg_monthly_return > 0:
                    monthly_returns[symbol] = avg_monthly_return
                    volume_weighted_prices[symbol] = volume_weighted_price

        sorted_symbols = [
            s for _, s in sorted(
                volume_weighted_prices.items(), key=lambda item: item[1], reverse=True
            ) if s in symbols
        ][: self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in sorted_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest