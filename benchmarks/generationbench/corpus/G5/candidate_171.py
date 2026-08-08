from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and information efficiency. "
        "Highly liquid stocks are more likely to be fairly priced and provide better diversification. "
        "This strategy screens for high liquidity based on traded value and applies equal weights among selected assets."
    )

    def __init__(self, min_traded_value: float = 10_000_000) -> None:
        self._min_traded_value = min_traded_value

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        traded_values = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").last().alias("last_price"),
                        pl.col("volume").sum().alias("total_volume"))
                   .with_columns(((pl.col("total_volume") * pl.col("last_price")).alias("traded_value")))
        )

        screened_symbols = traded_values.filter(
            (pl.col("traded_value") > self._min_traded_value)
        ).select("symbol")

        if screened_symbols.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_5_symbols = [str(symbol) for symbol in screened_symbols["symbol"].to_list()[:5]]
        weight = 1.0 / len(top_5_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_5_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest