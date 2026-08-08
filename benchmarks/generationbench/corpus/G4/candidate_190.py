from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalReturnStrategy(Strategy):
    rationale = (
        "Leveraging historical seasonality in the Indian equity market, this strategy "
        "exploits months with consistently high returns. By focusing on October to December, "
        "it aims to capture higher returns associated with monsoon-related agricultural activities"
        " and festive spending."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 12)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Extract relevant symbols
        symbols = view.symbols

        # Calculate monthly returns for each symbol
        monthly_returns = []
        for symbol in symbols:
            if symbol not in history.symbol.to_list():
                continue
            daily_closes = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "adj_close"
            ).with_columns(
                (pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1.0
            ).sort("session_date").head(self._window * 2)
            if daily_closes.height < self._window + 1:
                continue

            monthly_returns.append({
                "symbol": symbol,
                "returns": [float(v) for v in daily_closes["adj_close"].to_list()[-self._window:]]
            })

        # Identify top performing months
        top_months = _identify_top_months(monthly_returns, stamp.month)

        # Rank stocks based on their performance in top months
        ranked_stocks = _rank_stocks(top_months)
        if not ranked_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _identify_top_months(monthly_returns: list[dict], current_month: int) -> dict[str, float]:
    top_months = {}
    for month_return in monthly_returns:
        symbol = month_return["symbol"]
        returns = month_return["returns"]
        average_return = sum(returns) / len(returns)
        if current_month in [10, 11, 12] and average_return > 0.05:  # Example threshold
            top_months[symbol] = average_return

    return top_months


def _rank_stocks(top_months: dict[str, float]) -> list[str]:
    sorted_stocks = sorted(top_months.items(), key=lambda x: x[1], reverse=True)
    return [stock for stock, _ in sorted_stocks[:20]]  # Select top 20 stocks