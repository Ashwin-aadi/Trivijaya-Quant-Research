from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class HybridStrategy(Strategy):
    rationale = (
        "This strategy combines momentum, relative strength (RSI), volume-weighted average price (VWAP), and macroeconomic conditions through the Nifty 50 Index and patent filings to ensure robust stock selection across short-term trends, value signals, and broader market stability."
    )

    def __init__(self, momentum_window: int = 30, rsi_window: int = 14, vwap_lookback: int = 20, macro_threshold: float = 70) -> None:
        self._momentum_window = momentum_window
        self._rsi_window = rsi_window
        self._vwap_lookback = vwap_lookback
        self._macro_threshold = macro_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=120)

        if history.height < 120:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        vwap = _vwap(closes, self._vwap_lookback)
        momentum = _momentum(closes, self._momentum_window)
        rsi = _rsi(closes, self._rsi_window)
        macro = _macro_score(view)

        if macro < self._macro_threshold:
            return Signal(information_available_at=stamp, weights={})

        candidates: list[str] = []
        for symbol in view.symbols:
            if (symbol not in vwap.columns) or (symbol not in momentum.columns) or (symbol not in rsi.columns):
                continue
            vwap_val = float(vwap[symbol].max().item())
            mom_val = float(momentum[symbol][-1])
            rs_val = float(rsi[symbol][-1])

            if vwap_val > 0 and mom_val > 0 and rs_val >= 0:
                candidates.append(symbol)

        top_25 = candidates[:25]
        weights = {s: 4.0 / len(top_25) for s in top_25}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest


def _vwap(closes: pl.DataFrame, lookback: int) -> pl.DataFrame:
    weights = closes["volume"] / (closes.groupby("symbol").agg(pl.col("volume").sum()).with_columns(
        (pl.lit(1).alias("weight")).repeat_by(pl.col("volume"))
    )["weight"])
    vwap = closes.with_columns(weights).groupby("symbol").agg(
        ((pl.col("adj_close") * pl.col("weights")).sum() / pl.col("volume").sum()).alias("vwap")
    ).sort("session_date", descending=True)
    return vwap.tail(lookback)


def _momentum(closes: pl.DataFrame, window: int) -> pl.DataFrame:
    momentum = closes.groupby("symbol").agg(
        ((pl.col("adj_close") / pl.col("adj_close").shift(window) - 1.0).alias("mom"))
    ).sort("session_date", descending=True)
    return momentum.tail(window)


def _rsi(closes: pl.DataFrame, window: int) -> pl.DataFrame:
    changes = closes.with_columns(
        (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().alias("change")
    )
    gains = changes.filter(pl.col("change") > 0).with_columns(
        (pl.col("change")).alias("gain")
    ).groupby("symbol").agg(
        (pl.col("gain").sum()).alias("total_gain")
    )

    losses = changes.filter(pl.col("change") <= 0).with_columns(
        (-pl.col("change")).alias("loss")
    ).groupby("symbol").agg(
        (pl.col("loss").sum()).alias("total_loss")
    )

    avg_gain = gains.with_columns(
        ((pl.col("total_gain") / window).alias("avg_gain"))
    )
    avg_loss = losses.with_columns(
        ((pl.col("total_loss") / window).alias("avg_loss"))
    )

    rs = (avg_gain.join(avg_loss, on="symbol", how="left")).with_columns(
        ((pl.col("avg_gain") / pl.col("avg_loss").fill_null(1)).alias("rs"))
    ).groupby("symbol").agg(
        ((pl.col("rs").mean()).alias("rsi"))
    )
    return rs.sort("session_date", descending=True)


def _macro_score(view: MarketView) -> float:
    nifty = view.history().select(["session_date", "close"]).filter(pl.col("symbol") == "NIFTY 50")
    patent_filings = pl.DataFrame({"date": [date(2021, 1, 1), date(2021, 7, 1)], "count": [100, 150]})

    macro_nifty = nifty.join(patent_filings, on="date", how="inner")
    if macro_nifty.height < 2:
        return 0.0

    avg_close = float(macro_nifty["close"].mean().item())
    patent_change = (macro_nifty["count"][1] - macro_nifty["count"][0]) / macro_nifty["count"][0]

    score = (avg_close * 0.6 + patent_change * 0.4).clip(0, 1) * 100
    return float(score)