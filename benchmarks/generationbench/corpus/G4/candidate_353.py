from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "To exploit breakout continuation in the Indian market, we aim to capitalize on price movements after significant breakouts. "
        "Breakouts often indicate a shift in market sentiment and momentum, which can be sustained if confirmed by further price action."
    )

    def __init__(self, lookback_period: int = 30, retrace_days: int = 5, volume_threshold: float = 1.5) -> None:
        self._lookback_period = lookback_period
        self._retrace_days = retrace_days
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period + 1)
        if history.height < self._lookback_period + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback_period)

        breakout_candidates: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in history["symbol"].to_list():
                continue
            hh, ll = _calculate_hh_ll(history.filter(pl.col("symbol") == symbol))
            if not hh or not ll:
                continue

            close_values = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].drop_nulls().to_list()]
            highest_high, lowest_low = max(close_values), min(close_values)
            breakout_price = highest_high if highest_high > ll else lowest_low
            breakout_confirmation = _check_breakout_confirmation(
                history.filter(pl.col("symbol") == symbol),
                hh,
                ll,
                breakout_price,
                self._volume_threshold,
            )

            if breakout_confirmation:
                retrace_window = history.filter(pl.col("symbol") == symbol).tail(self._retrace_days)
                retrace_confirmation = _check_retrace(
                    retrace_window["adj_close"].to_list(),
                    retrace_window["session_date"].to_list(),
                    close_values,
                )
                if retrace_confirmation:
                    breakout_candidates.append(symbol)

        top_n_symbols = sorted(breakout_candidates, key=lambda x: -1 * closes[x].sum())[:10]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_hh_ll(history: pl.DataFrame) -> tuple[float | None, float | None]:
    hh = history.filter(pl.col("adj_close") == history.select(pl.col("adj_close").max()).item())
    ll = history.filter(pl.col("adj_close") == history.select(pl.col("adj_close").min()).item())
    return (hh["adj_close"].last().item() if not hh.is_empty() else None,
            ll["adj_close"].last().item() if not ll.is_empty() else None)


def _check_breakout_confirmation(history: pl.DataFrame, hh: float | None, ll: float | None, breakout_price: float, volume_threshold: float) -> bool:
    confirmation = False
    for i in range(1, history.height):
        row = history.row(i)
        symbol, session_date, adj_close, volume = row[0], row[1], row[4], row[5]
        if not (hh or ll):
            return False
        if breakout_price > hh:
            confirmation = adj_close < hh and (volume / pl.col("volume").mean().item()) >= volume_threshold
        elif breakout_price < ll:
            confirmation = adj_close > ll and (volume / pl.col("volume").mean().item()) >= volume_threshold
        if confirmation:
            break
    return confirmation


def _check_retrace(prices: list[float], dates: list[date], close_values: list[float]) -> bool:
    for i, price in enumerate(prices):
        retrace_price = closest_close(close_values, date_to_index(dates[i]))
        if abs(price - retrace_price) / max(close_values) < 0.1:
            return True
    return False


def date_to_index(date: date) -> int:
    return view.history().filter(pl.col("session_date") == date)["adj_close"].row(0)[0]


def closest_close(prices: list[float], index: int) -> float:
    return sorted(prices, key=lambda x: abs(x - prices[index]))[1]