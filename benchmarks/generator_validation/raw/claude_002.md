## Cross-Sectional Momentum

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Stocks that have outperformed their peers over the past six months tend to keep "
        "outperforming over the following weeks, because investors under-react to slow-moving "
        "fundamental news and institutional flows arrive in stages rather than all at once. "
        "The most recent month is deliberately skipped, since returns at that horizon reverse "
        "and would otherwise contaminate the ranking. Equal weighting across the strongest "
        "names keeps the bet on the cross-sectional ranking itself rather than on any single "
        "company."
    )

    def __init__(self, lookback: int = 126, skip: int = 21, top_n: int = 10) -> None:
        self._lookback = lookback
        self._skip = skip
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._lookback + self._skip
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            score = _window_return(values, self._lookback, self._skip)
            if score is not None:
                scores[symbol] = score

        picks = _top_symbols(scores, self._top_n)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _window_return(values: list[float], lookback: int, skip: int) -> float | None:
    start = values[-(lookback + skip)]
    end = values[-1 - skip] if skip > 0 else values[-1]
    if start <= 0.0:
        return None
    return end / start - 1.0


def _top_symbols(scores: dict[str, float], top_n: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, score in ranked[:top_n] if score > 0.0]


def _latest_visible(view: MarketView) -> date:
    visible: pl.DataFrame = view.history()
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
        "Over a few days, price moves in liquid large caps are driven more by impatient order "
        "flow than by information, and the liquidity providers who absorb that flow demand "
        "compensation which shows up as a bounce. Buying the weakest names of the last week is "
        "a way of collecting that compensation. A floor on the permitted decline filters out "
        "genuine repricing events such as guidance cuts, where the drop reflects new "
        "information and no reversal should be expected."
    )

    def __init__(
        self, window: int = 5, bottom_n: int = 10, floor: float = -0.25
    ) -> None:
        self._window = window
        self._bottom_n = bottom_n
        self._floor = floor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._window + 1
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            start = values[-need]
            if start <= 0.0:
                continue
            change = values[-1] / start - 1.0
            if self._floor <= change < 0.0:
                scores[symbol] = change

        picks = _weakest(scores, self._bottom_n)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _weakest(scores: dict[str, float], bottom_n: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1])
    return [symbol for symbol, _ in ranked[:bottom_n]]


def _latest_visible(view: MarketView) -> date:
    visible: pl.DataFrame = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Volatility-Scaled Trend Following

```python
from __future__ import annotations

import math
from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "A price above its long moving average is the simplest evidence that a repricing is "
        "still under way, and such trends persist long enough to be traded on a daily "
        "schedule. Sizing each position inversely to its own realised volatility means a "
        "sleepy consumer name and a jumpy metals name contribute similar amounts of risk, so "
        "the portfolio is not silently dominated by whichever sector happens to be turbulent. "
        "A per-name cap and a gross cap stop the inverse-volatility term from concentrating "
        "the book in a handful of unusually calm stocks."
    )

    def __init__(
        self,
        trend_window: int = 100,
        vol_window: int = 20,
        target_vol: float = 0.20,
        max_weight: float = 0.10,
        gross_cap: float = 1.0,
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._target_vol = target_vol
        self._max_weight = max_weight
        self._gross_cap = gross_cap

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = max(self._trend_window, self._vol_window + 1)
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        raw: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            weight = self._size(values)
            if weight > 0.0:
                raw[symbol] = weight

        if not raw:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights=_apply_gross_cap(raw, self._gross_cap),
        )

    def _size(self, values: list[float]) -> float:
        trend = values[-self._trend_window :]
        average = sum(trend) / len(trend)
        if values[-1] <= average:
            return 0.0
        rets = _returns(values[-(self._vol_window + 1) :])
        vol = _stdev(rets) * math.sqrt(252.0)
        if vol <= 0.0:
            return 0.0
        return min(self._target_vol / vol, self._max_weight)


def _returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for prev, curr in zip(values[:-1], values[1:]):
        if prev > 0.0:
            out.append(curr / prev - 1.0)
    return out


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _apply_gross_cap(raw: dict[str, float], gross_cap: float) -> dict[str, float]:
    total = sum(raw.values())
    if total <= gross_cap:
        return raw
    scale = gross_cap / total
    return {symbol: weight * scale for symbol, weight in raw.items()}


def _latest_visible(view: MarketView) -> date:
    visible: pl.DataFrame = view.history()
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
        "Equal weighting is a robust default because it avoids the estimation error that ruins "
        "optimised portfolios, but it only survives contact with reality if every holding can "
        "actually be traded at the modelled price. Screening on median daily traded value, "
        "rather than mean, keeps a single block deal from promoting an otherwise thin name into "
        "the book. Requiring a name to have printed on most sessions in the window also drops "
        "recent listings and suspended stocks whose backtest prices are misleading. What "
        "remains is a tradable, low-turnover core of the index."
    )

    def __init__(
        self, window: int = 60, top_n: int = 25, min_coverage: float = 0.8
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._min_coverage = min_coverage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        frame = view.history(lookback=self._window)
        if frame.is_empty():
            return Signal(information_available_at=stamp, weights={})
        sessions = len(set(frame["session_date"].to_list()))
        if sessions < self._window:
            return Signal(information_available_at=stamp, weights={})

        buckets = _turnover_buckets(frame, set(view.symbols))
        minimum = int(self._min_coverage * self._window)
        scores: dict[str, float] = {}
        for symbol, values in buckets.items():
            if len(values) >= minimum:
                scores[symbol] = _median(values)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        picks = [symbol for symbol, _ in ranked[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _turnover_buckets(
    frame: pl.DataFrame, universe: set[str]
) -> dict[str, list[float]]:
    symbols = frame["symbol"].to_list()
    closes = frame["close"].to_list()
    volumes = frame["volume"].to_list()
    buckets: dict[str, list[float]] = {}
    for symbol, close, volume in zip(symbols, closes, volumes):
        if symbol not in universe or close is None or volume is None:
            continue
        turnover = float(close) * float(volume)
        if turnover > 0.0:
            buckets.setdefault(str(symbol), []).append(turnover)
    return buckets


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _latest_visible(view: MarketView) -> date:
    visible: pl.DataFrame = view.history()
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


class TrailingReferenceReversion(Strategy):
    rationale = (
        "A quarter of closing prices is a reasonable proxy for the level at which the marginal "
        "long-term holder of a large cap was willing to transact, and prices that wander well "
        "below it without a change in fundamentals tend to be pulled back toward it as value "
        "buyers step in. The reference is computed strictly from sessions before the current "
        "one, so the level being compared against is not partly made of the price under "
        "evaluation. A minimum gap requirement means the strategy sits in cash rather than "
        "trading noise when nothing has actually dislocated."
    )

    def __init__(
        self, window: int = 60, top_n: int = 10, min_gap: float = 0.05
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._min_gap = min_gap

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        need = self._window + 1
        closes = view.closes(lookback=need)
        if closes.height < need:
            return Signal(information_available_at=stamp, weights={})

        gaps: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < need:
                continue
            gap = _gap_to_reference(values, self._window)
            if gap is not None and gap <= -self._min_gap:
                gaps[symbol] = gap

        ranked = sorted(gaps.items(), key=lambda item: item[1])
        picks = [symbol for symbol, _ in ranked[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _gap_to_reference(values: list[float], window: int) -> float | None:
    prior = values[-(window + 1) : -1]
    if len(prior) < window:
        return None
    reference = sum(prior) / len(prior)
    if reference <= 0.0:
        return None
    return values[-1] / reference - 1.0


def _latest_visible(view: MarketView) -> date:
    visible: pl.DataFrame = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```
