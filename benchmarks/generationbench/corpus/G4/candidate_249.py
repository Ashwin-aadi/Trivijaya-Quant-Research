from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Lower volatility stocks have historically shown higher average returns due to risk "
        "premium compensation and market inefficiencies. This strategy selects the bottom decile "
        "of stocks based on their 12-month realized volatility for inclusion in the portfolio, "
        "aiming to reduce overall portfolio risk while potentially enhancing returns."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_dict()["symbol"]:
                continue
            daily_returns = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
                )
                .sort("session_date", descending=False)
                .drop_nulls()
            )

            if daily_returns.height < self._window:
                continue
            realized_volatility = (
                daily_returns.select(pl.col("r").std().alias("vol"))
                .with_columns(
                    (pl.col("vol") * 252**0.5).alias("ann_vol")
                )
                .select("ann_vol")
                .item()
            )

            if len(picks) < self._top_n:
                picks.append(symbol)
            else:
                current_max = max([pl.col(s).item() for s in picks], default=9999.0)
                if realized_volatility < current_max:
                    picks[picks.index(min(picks, key=lambda x: pl.col(x).item()))] = symbol

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest