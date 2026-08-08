from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets with higher relative strength (i.e., those that have outperformed the market "
        "over a given period) are likely to continue their strong performance. This strategy "
        "buys assets in the top decile of relative strength."
    )

    def __init__(self, window: int = 20, num_top_assets: int = 10) -> None:
        self._window = window
        self._num_top_assets = num_top_assets

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each asset
        prices = [float(v) for v in closes.to_dict().values()]
        returns = [(p[-1] - p[0]) / p[0] for p in zip(*prices)]

        # Calculate relative strength by ranking assets based on their returns
        rank_df = pl.DataFrame({"symbol": view.symbols, "return": returns})
        rank_df = rank_df.with_columns(
            (pl.col("return").rank(method="ordinal", descending=True)).alias("rs_rank")
        )
        
        top_assets = rank_df.sort("rs_rank").head(self._num_top_assets)["symbol"].to_list()

        if not top_assets:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_assets)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_assets}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest