from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with outperforming relative strength against the NIFTY 100 index. "
        "By ranking stocks based on their relative performance and rebalancing periodically, we aim to "
        "capitalize on persistent strong performers while reducing exposure to underperformers."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5, top_n: int = 30) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        index_history = (
            history.select("session_date", "adj_close").filter(pl.col("symbol") == "NIFTY 100")
        )
        stock_history = history.filter(pl.col("symbol").is_in(view.symbols))

        # Calculate daily returns
        index_returns = (index_history["adj_close"].shift(-1) / index_history["adj_close"] - 1.0).alias(
            "index_return"
        )
        stock_returns = (
            stock_history.with_columns(
                (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0).alias("stock_return")
            ).select("symbol", "session_date", "stock_return")
        )

        # Compute relative strength ratio
        merged_data = (
            stock_returns.join(index_returns, on=["session_date"], how="left")
            .with_columns(
                (pl.col("stock_return") / pl.col("index_return")).alias("relative_strength_ratio"),
                (pl.col("symbol").cast(pl.Categorical)).arr.to_list().alias("symbols"),
            )
        )

        if merged_data.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank by relative strength ratio
        ranked_data = (
            merged_data.sort(
                "relative_strength_ratio", descending=True
            ).group_by(["symbols"]).agg(
                pl.col("relative_strength_ratio").mean().alias("avg_ratio")
            )
        )

        top_symbols = [s for s in ranked_data["symbols"].to_list()[0][: self._top_n]]
        weights = {symbol: 1.0 / len(top_symbols) for symbol in top_symbols}

        return Signal(
            information_available_at=stamp, weights={k: float(v) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest