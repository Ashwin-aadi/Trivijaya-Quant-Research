from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalReturnsStrategy(Strategy):
    rationale = (
        "Exploiting month-of-year seasonality in the Indian equity market by focusing on "
        "historical performance and ranking stocks based on their past returns. "
        "Long positions are taken in sectors expected to outperform during favorable months, "
        "while short positions are initiated in defensive sectors likely to underperform."
    )

    def __init__(self, window: int = 5 * 252, top_n_long: int = 30, bottom_n_defensive: int = 10) -> None:
        self._window = window
        self._top_n_long = top_n_long
        self._bottom_n_defensive = bottom_n_defensive

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < 252 * 5:
            return Signal(information_available_at=stamp, weights={})

        monthly_returns = self._calculate_monthly_returns(history)

        top_decile_long = _select_top_deciqle(monthly_returns, "long")
        bottom_decile_defensive = _select_bottom_deciqle(monthly_returns, "defensive")

        weights = {}
        for symbol in top_decile_long:
            if symbol not in view.symbols:
                continue
            weights[symbol] = 1.0 / len(top_decile_long)

        for symbol in bottom_decile_defensive:
            if symbol not in view.symbols:
                continue
            weights[symbol] = -1.0 / len(bottom_decile_defensive)

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest


def _calculate_monthly_returns(history: pl.DataFrame) -> pl.DataFrame:
    monthly_hist = history.with_columns(
        [
            (pl.col("close") / pl.col("open").shift(252) - 1.0).alias("monthly_return"),
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
        ]
    ).select(
        [pl.col("symbol"), "session_date", "monthly_return"]
    )

    monthly_hist = monthly_hist.group_by(pl.col("symbol")).agg([
        (pl.col("monthly_return").mean().alias("avg_monthly_return")),
        (pl.col("monthly_return").std().alias("std_dev"))
    ])

    return monthly_hist


def _select_top_deciqle(monthly_returns: pl.DataFrame, category: str) -> list[str]:
    if category == "long":
        top_decile = monthly_returns.sort(pl.col("avg_monthly_return"), descending=True).head(int(len(monthly_returns) * 0.1))
    elif category == "defensive":
        bottom_decile = monthly_returns.sort(pl.col("avg_monthly_return")).head(int(len(monthly_returns) * 0.1))
    else:
        raise ValueError("Invalid category: must be 'long' or 'defensive'")
    
    return top_decile["symbol"].to_list()