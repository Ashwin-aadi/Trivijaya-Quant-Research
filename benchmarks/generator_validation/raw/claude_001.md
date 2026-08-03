## Cross-Sectional Momentum

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Relative strength persists over intermediate horizons because information about "
        "earnings and demand diffuses slowly through a market and investors anchor on stale "
        "valuations, so past winners keep outrunning past losers for months. The most recent "
        "few weeks are skipped because microstructure effects and short-term reversal "
        "contaminate the freshest part of the window. Ranking cross-sectionally rather than "
        "against zero keeps the book market-neutral in spirit and lets the signal work in "
        "both rising and falling regimes."
    )

    def __init__(self, lookback: int = 126, skip: int = 21, top_n: int = 10) -> None:
        self._lookback = lookback
        self._skip = skip
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._lookback + 1
        closes = view.closes(lookback=need)
        if closes.height < need or self._skip >= self._lookback:
            return Signal(information_available_at=stamp, weights={})

        scored = self._score_universe(view, closes, need)
        if not scored:
            return Signal(information_available_at=stamp, weights={})

        scored.sort(key=lambda pair: pair[0], reverse=True)
        picks = [symbol for _, symbol in scored[: self._top_n]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _score_universe(
        self, view: MarketView, closes: pl.DataFrame, need: int
    ) -> list[tuple[float, str]]:
        scored: list[tuple[float, str]] = []
        for symbol in view.symbols:
            values = _column_values(closes, symbol)
            if len(values) < need:
                continue
            start = values[-need]
            end = values[-(self._skip + 1)]
            if start <= 0.0:
                continue
            scored.append((end / start - 1.0, symbol))
        return scored


def _column_values(frame: pl.DataFrame, symbol: str) -> list[float]:
    if symbol not in frame.columns:
        return []
    return [float(v) for v in frame[symbol].drop_nulls().to_list()]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Short-Horizon Mean Reversion

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ShortHorizonReversal(Strategy):
    rationale = (
        "Over a handful of sessions price moves are dominated by liquidity demand rather than "
        "news: a fund unwinding a position pushes a stock below fair value and pays the "
        "spread to whoever will take the other side. Buying the sharpest recent decliners is "
        "the systematic way to supply that liquidity and collect the reversal premium. The "
        "effect decays within days, so the holding horizon must be short and the entry "
        "confined to names that actually fell rather than merely lagged."
    )

    def __init__(
        self, window: int = 5, bottom_n: int = 10, max_drop: float = -0.02
    ) -> None:
        self._window = window
        self._bottom_n = bottom_n
        self._max_drop = max_drop

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._window + 1
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        losers = self._rank_losers(view, closes, need)
        if not losers:
            return Signal(information_available_at=stamp, weights={})

        losers.sort(key=lambda pair: pair[0])
        picks = [symbol for _, symbol in losers[: self._bottom_n]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _rank_losers(
        self, view: MarketView, closes: pl.DataFrame, need: int
    ) -> list[tuple[float, str]]:
        ranked: list[tuple[float, str]] = []
        for symbol in view.symbols:
            values = _column_values(closes, symbol)
            if len(values) < need:
                continue
            base = values[-need]
            if base <= 0.0:
                continue
            move = values[-1] / base - 1.0
            if move <= self._max_drop:
                ranked.append((move, symbol))
        return ranked


def _column_values(frame: pl.DataFrame, symbol: str) -> list[float]:
    if symbol not in frame.columns:
        return []
    return [float(v) for v in frame[symbol].drop_nulls().to_list()]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Volatility-Scaled Trend Following

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "A price sustained above its long moving average identifies a name whose drift is "
        "positive, and trends persist because investors underreact to slow-moving changes in "
        "fundamentals before eventually chasing them. Sizing each position inversely to its "
        "own realised volatility equalises the risk each name contributes, so a single "
        "high-beta stock cannot dominate the book. Because gross exposure falls automatically "
        "when volatility rises across the market, the strategy de-risks into turbulence "
        "instead of holding a constant notional through a drawdown."
    )

    def __init__(
        self,
        trend_window: int = 100,
        vol_window: int = 20,
        target_vol: float = 0.20,
        max_weight: float = 0.10,
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._target_vol = target_vol
        self._max_weight = max_weight

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = max(self._trend_window, self._vol_window + 1)
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        raw = self._raw_weights(view, closes, need)
        if not raw:
            return Signal(information_available_at=stamp, weights={})

        gross = sum(raw.values())
        scale = 1.0 / gross if gross > 1.0 else 1.0
        weights = {s: w * scale for s, w in raw.items()}
        return Signal(information_available_at=stamp, weights=weights)

    def _raw_weights(
        self, view: MarketView, closes: pl.DataFrame, need: int
    ) -> dict[str, float]:
        raw: dict[str, float] = {}
        for symbol in view.symbols:
            values = _column_values(closes, symbol)
            if len(values) < need:
                continue
            trend = values[-self._trend_window :]
            if values[-1] <= sum(trend) / len(trend):
                continue
            vol = _annualised_vol(values[-(self._vol_window + 1) :])
            if vol <= 0.0:
                continue
            raw[symbol] = min(self._target_vol / vol, self._max_weight)
        return raw


def _annualised_vol(values: list[float]) -> float:
    returns: list[float] = []
    for previous, current in zip(values[:-1], values[1:]):
        if previous > 0.0:
            returns.append(current / previous - 1.0)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return (variance**0.5) * (252.0**0.5)


def _column_values(frame: pl.DataFrame, symbol: str) -> list[float]:
    if symbol not in frame.columns:
        return []
    return [float(v) for v in frame[symbol].drop_nulls().to_list()]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Liquidity-Screened Equal Weighting

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeight(Strategy):
    rationale = (
        "Equal weighting is a robust prior when no return forecast is trusted, since it makes "
        "no bet on capitalisation and rebalances mechanically out of names that have run up. "
        "Its practical weakness is that a naive equal-weight book puts as much money into the "
        "thinnest constituent as into the deepest one, and implementation shortfall then eats "
        "the diversification benefit. Screening on median traded value rather than mean "
        "avoids letting one block deal qualify an otherwise illiquid stock, so the surviving "
        "book is the part of the index that can actually be traded at scale."
    )

    def __init__(self, window: int = 60, top_n: int = 25, min_sessions: int = 40) -> None:
        self._window = window
        self._top_n = top_n
        self._min_sessions = min_sessions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        frame = view.history(lookback=self._window)
        if frame.is_empty():
            return Signal(information_available_at=stamp, weights={})

        turnover = _turnover_by_symbol(frame)
        ranked = self._rank_liquidity(view, turnover)
        if not ranked:
            return Signal(information_available_at=stamp, weights={})

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        picks = [symbol for _, symbol in ranked[: self._top_n]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _rank_liquidity(
        self, view: MarketView, turnover: dict[str, list[float]]
    ) -> list[tuple[float, str]]:
        ranked: list[tuple[float, str]] = []
        for symbol in view.symbols:
            series = turnover.get(symbol, [])
            if len(series) < self._min_sessions:
                continue
            level = _median(series)
            if level <= 0.0:
                continue
            ranked.append((level, symbol))
        return ranked


def _turnover_by_symbol(frame: pl.DataFrame) -> dict[str, list[float]]:
    ordered = frame.sort("session_date")
    symbols = ordered["symbol"].to_list()
    prices = ordered["close"].to_list()
    volumes = ordered["volume"].to_list()
    out: dict[str, list[float]] = {}
    for symbol, price, volume in zip(symbols, prices, volumes):
        if symbol is None or price is None or volume is None:
            continue
        out.setdefault(str(symbol), []).append(float(price) * float(volume))
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    size = len(ordered)
    if size == 0:
        return 0.0
    middle = size // 2
    if size % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Price-Level Reversion Against a Trailing Reference

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingAnchorReversion(Strategy):
    rationale = (
        "A trailing average price acts as a slow-moving reference for what a stock has "
        "recently been worth, and a large negative gap to that reference marks a dislocation "
        "rather than a repriced fundamental in most cases. Buying the widest discounts is a "
        "bet that the anchor pulls price back rather than price dragging the anchor down. A "
        "floor on the gap is essential because a truly broken name gaps far below its "
        "reference and stays there, so only moderate dislocations are treated as tradeable."
    )

    def __init__(
        self,
        window: int = 60,
        top_n: int = 10,
        entry_gap: float = -0.05,
        floor_gap: float = -0.35,
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._entry_gap = entry_gap
        self._floor_gap = floor_gap

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        candidates = self._gaps(view, closes)
        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        candidates.sort(key=lambda pair: pair[0])
        picks = [symbol for _, symbol in candidates[: self._top_n]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

    def _gaps(self, view: MarketView, closes: pl.DataFrame) -> list[tuple[float, str]]:
        found: list[tuple[float, str]] = []
        for symbol in view.symbols:
            values = _column_values(closes, symbol)
            if len(values) < self._window:
                continue
            window = values[-self._window :]
            reference = sum(window) / len(window)
            if reference <= 0.0:
                continue
            gap = values[-1] / reference - 1.0
            if self._floor_gap <= gap <= self._entry_gap:
                found.append((gap, symbol))
        return found


def _column_values(frame: pl.DataFrame, symbol: str) -> list[float]:
    if symbol not in frame.columns:
        return []
    return [float(v) for v in frame[symbol].drop_nulls().to_list()]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```
