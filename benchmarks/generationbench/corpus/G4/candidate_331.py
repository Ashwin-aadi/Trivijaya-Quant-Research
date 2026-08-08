from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "To exploit periods of range compression following high dispersion in India's equity market, "
        "we identify stocks with low current volatility but a recent increase. These stocks are expected to "
        "benefit from increased volatility as the market reestablishes equilibrium."
    )

    def __init__(self, lookback: int = 30, top_n: int = 20) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns
        history = history.with_columns(
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Calculate historical volatility (HV)
        hv = (
            history.groupby("symbol")
            .agg((pl.col("return").std() * (252 ** 0.5)).alias("hv"))
            .sort("hv", descending=False)
            .to_pandas()
        )
        
        # Rank symbols by HV
        ranks = hv["hv"].rank(method="dense", ascending=True).astype(int)
        ranked_symbols = list(zip(hv.index, ranks))
        ranked_symbols.sort(key=lambda x: x[1])
        
        # Select top N symbols with low current volatility and recent increase in volatility
        candidates = []
        for symbol, rank in ranked_symbols:
            if rank <= self._top_n:
                recent_hv = history.filter(pl.col("symbol") == symbol)[-self._lookback:]
                recent_hv_increase = (recent_hv["return"].std() > 0.01) and (
                    recent_hv["return"][-1] > recent_hv["return"][:-1].mean()
                )
                if recent_hv_increase:
                    candidates.append(symbol)
        
        # Assign weights
        weight = 1.0 / len(candidates) if candidates else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest