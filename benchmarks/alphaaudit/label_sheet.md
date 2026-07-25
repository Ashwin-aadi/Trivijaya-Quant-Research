# Semantic audit — hand-labelling sheet

Fifty candidate strategies. Each item gives an author's stated economic rationale and the code that
implements it. Assign exactly one label per item, judging only whether the rationale is an honest
description of what the code does.

These same fifty items are put to the local model that forms the semantic audit layer. Cohen's
kappa between your labels and its labels is the measurement, so your labels must be formed
independently: work through the sheet in order and do not consult the model's output, the source
files, or the CSV's provenance columns until you have finished.

## The four labels

- **rationale_implementation_mismatch** — the code does not do what the rationale claims (a
  different window, a different direction, a rule that is described but never applied).
- **unfalsifiable_mechanism** — the rationale is phrased so that no outcome could contradict it; it
  would explain a profit and a loss equally well.
- **unacknowledged_known_anomaly** — a long-documented effect (momentum, low volatility, size,
  value, short-term reversal) is presented as novel, proprietary, or newly discovered. Implementing
  a known effect is not a defect; claiming to have found it is.
- **consistent** — none of the above. The rationale describes what the code does. It may still be
  wrong about the market; that is not the question here.

## Rules

1. Judge the code as written. Do not speculate about what the author meant.
2. If you are unsure, answer `consistent`.
3. Being simple, unoriginal, or unprofitable is not a defect.
4. If more than one label applies, use the first that applies in the order listed above.

## What you are shown

Each excerpt is the strategy class: its docstring and its methods, with imports and the module
docstring left out. The class's own `rationale` attribute is removed from every excerpt, so the
rationale printed above the code is the only one in play. Where a class was too long to print
whole, the excerpt is cut to the docstring and `generate`, and a comment inside the code block
names what was left out.

Write your answer on the `Your label:` line, then transcribe the fifty labels into the
`human_label` column of `label_sheet.csv`, matching on the item id.

---

### Item 01 — `itm-4648f68e`

**Rationale**

A short average above a long one indicates recent prices are running ahead of the established level, which is the conventional definition of an uptrend. The portfolio holds only names in that state and stays out of the rest.

**Code**

```python
class SmaCrossover(Strategy):
    """Classic trend filter on two simple moving averages."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._long)
        if closes.height < self._long:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._long:
                continue
            short_avg = sum(values[-self._short:]) / self._short
            long_avg = sum(values[-self._long:]) / self._long
            if short_avg > long_avg:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 02 — `itm-3282b974`

**Rationale**

Day-of-week effects are among the most-tested and least-durable regularities in equity returns, and any that survive are usually too small to clear costs. This holds the whole universe when the most recent completed session fell on a Monday or a Tuesday and sits in cash otherwise. There is no mechanism behind it beyond the historical record, so it is expected to fail; it is here as a weak-signal reference point.

**Code**

```python
class SeasonalityDayOfWeek(Strategy):
    """A deliberately weak calendar rule, included to measure what a null looks like."""

    def __init__(self, weekdays: tuple[int, ...] = DEFAULT_WEEKDAYS) -> None:
        self._weekdays = weekdays

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        if not view.symbols or view.history(lookback=1).is_empty():
            return Signal(information_available_at=stamp, weights={})

        # The weekday tested is that of the last completed session, so the position is carried
        # over the session that follows it. Reading the traded session's own calendar date would
        # mean reaching past the decision moment, which this class never does.
        if stamp.weekday() not in self._weekdays:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(sorted(view.symbols)),
        )
```

Your label: ______________________________

---

### Item 03 — `itm-6c0f0fbe`

**Rationale**

A random portfolio is the honest null: any strategy that cannot beat one has demonstrated no skill. It also exposes how much of a backtest's spread comes from selection versus from the market itself.

**Code**

```python
class RandomWalkBaseline(Strategy):
    """Selects a random subset each rebalance, reproducibly."""

    def __init__(self, holdings: int = 10, seed: int = SEED) -> None:
        self._holdings = holdings
        self._seed = seed

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        candidates = sorted(view.symbols)
        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        # Seeded per decision date rather than once at construction, so the sequence of picks does
        # not depend on how many times the strategy has previously been called. The same date
        # always yields the same portfolio, whatever order the backtest runs in.
        rng = random.Random(f"{self._seed}-{stamp.isoformat()}")
        count = min(self._holdings, len(candidates))
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(rng.sample(candidates, count)),
        )
```

Your label: ______________________________

---

### Item 04 — `itm-d11773ab`

**Rationale**

At horizons of three years and beyond, equity returns have historically reversed rather than continued, the usual explanation being that a long run of bad news drives valuation further than the fundamentals justify. This buys the worst three-year performers, which is deliberately the opposite sign to the shorter-horizon momentum rules in this set.

**Code**

```python
class LongTermReversal756d(Strategy):
    """Long-horizon reversal: the opposite sign to the short-horizon momentum rules."""

    def __init__(self, lookback: int = 756, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        # Three years of history must genuinely exist. Without this guard window_return would
        # quietly measure whatever shorter span is available and the strategy would stop being a
        # long-horizon rule during the early part of any backtest.
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        returns = window_return(view, self._lookback)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(returns, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---

### Item 05 — `itm-510de326`

**Rationale**

Demanding agreement between a one-month and a six-month window filters out names whose recent strength contradicts their longer trend, which is a common way to reduce the whipsaw a single-window momentum rule suffers.

**Code**

```python
class DualMomentum21x126(Strategy):
    """Holds only names trending up on both horizons."""

    def __init__(self, short_lookback: int = 21, long_lookback: int = 126) -> None:
        self._short = short_lookback
        self._long = long_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        short_returns = window_return(view, self._short)
        long_returns = window_return(view, self._long)
        picks = sorted(
            symbol for symbol in short_returns
            if short_returns[symbol] > 0 and long_returns.get(symbol, -1.0) > 0
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 06 — `itm-4621a2a0`

**Rationale**

Ranking on the trailing year while discarding the most recent month is a construction we arrived at internally and have not seen described anywhere. Dropping that month removes a short-horizon effect working against the longer one, and the spread that remains is the core of the edge here. It is not available in the published literature.

**Code**

```python
class MomentumSkipMonth(Strategy):
    """The conventional 12-1 momentum construction."""

    def __init__(self, lookback: int = 252, skip: int = 21, holdings: int = 10) -> None:
        if skip >= lookback:
            raise ValueError("the skipped window must be shorter than the lookback")
        self._lookback = lookback
        self._skip = skip
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback + 1:
                continue
            start = values[0]
            # Indexed back from the end of the visible window, so the last `skip` sessions are
            # excluded from the measurement while remaining strictly in the past.
            end = values[-1 - self._skip]
            if start <= 0:
                continue
            scores[symbol] = end / start - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 07 — `itm-60bddd44`

**Rationale**

Counting positive sessions describes how consistently a name has advanced, which is not the same as how far it advanced: one enormous day can dominate a total return without indicating any persistent trend. This ranks on consistency instead.

**Code**

```python
class TrendPersistence(Strategy):
    """Counts up-days rather than measuring total return."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        scores = {
            symbol: sum(1 for r in rets if r > 0) / len(rets)
            for symbol, rets in series.items()
            if rets
        }
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 08 — `itm-a3df6b52`

**Rationale**

Included as the mirror image of the low-volatility rule. If the low-volatility fixture shows an effect, this one should show its opposite; if both look similar, the apparent effect is more likely an artifact of the test setup than a property of the market.

**Code**

```python
class HighVolatility(Strategy):
    """Selects on realised volatility, highest first."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 09 — `itm-a06445d0`

**Rationale**

A price move on heavy turnover reflects more participants agreeing on a revaluation than the same move on a quiet day. Scaling momentum by traded value therefore favours trends with genuine weight behind them.

**Code**

```python
class VolumeWeightedMomentum(Strategy):
    """Momentum, discounted where turnover is thin."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        history = view.history(lookback=self._lookback)
        if not returns or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = history.group_by("symbol").agg(
            pl.col("turnover_inr").median().alias("median_turnover")
        )
        turnover = dict(
            zip(liquidity["symbol"].to_list(),
                liquidity["median_turnover"].to_list(), strict=True)
        )
        largest = max(turnover.values()) if turnover else 0.0
        if largest <= 0:
            return Signal(information_available_at=stamp, weights={})

        # Turnover is normalised to the largest name, so the scale factor sits in [0, 1] and the
        # ranking stays driven by the return rather than by absolute size.
        scores = {
            symbol: value * (turnover.get(symbol, 0.0) / largest)
            for symbol, value in returns.items()
        }
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 10 — `itm-fd159a00`

**Rationale**

Cross-sectional momentum is conventionally measured over twelve months with the most recent month dropped, because at the one-month horizon returns tend to reverse rather than continue. Including that month nets a reversal effect against a continuation effect and weakens both; skipping it measures the longer trend on its own.

**Code**

```python
class MomentumSkipMonth(Strategy):
    """The conventional 12-1 momentum construction."""

    def __init__(self, lookback: int = 252, skip: int = 21, holdings: int = 10) -> None:
        if skip >= lookback:
            raise ValueError("the skipped window must be shorter than the lookback")
        self._lookback = lookback
        self._skip = skip
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback + 1:
                continue
            start = values[0]
            # Indexed back from the end of the visible window, so the last `skip` sessions are
            # excluded from the measurement while remaining strictly in the past.
            end = values[-1 - self._skip]
            if start <= 0:
                continue
            scores[symbol] = end / start - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 11 — `itm-765b64f1`

**Rationale**

Cross-sectional momentum: names with strong recent returns continue to outperform over short horizons, so the portfolio holds the strongest performers each session.

**Code**

```python
class LeakFutureReturn(Strategy):
    """Selects names by the return of the session it is about to trade."""

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        # Holding the full panel is what makes the leak possible. A clean strategy never needs it.
        self._panel = panel
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        # THE CHEAT: read the fill session itself, which has not happened yet at decision time.
        future = self._panel.filter(
            (pl.col("session_date") == view.as_of)
            & (pl.col("symbol").is_in(list(view.symbols)))
            & (pl.col("adj_open") > 0)
        )
        if future.is_empty():
            return Signal(information_available_at=_previous_session(view), weights={})

        ranked = (
            future.with_columns(
                (pl.col("adj_close") / pl.col("adj_open") - 1.0).alias("future_return")
            )
            .sort("future_return", descending=True)
            .head(self._top_n)
        )
        chosen = ranked["symbol"].to_list()
        if not chosen:
            return Signal(information_available_at=_previous_session(view), weights={})

        weight = 1.0 / len(chosen)
        # The timestamp is honest even though the data is not — which is the point. A stamp check
        # alone cannot catch this; only reading the code reveals the leak.
        return Signal(
            information_available_at=_previous_session(view),
            weights=dict.fromkeys(chosen, weight),
        )
```

Your label: ______________________________

---

### Item 12 — `itm-4ede3cd7`

**Rationale**

Measuring a stock against its own universe rather than against zero removes the market move common to all of them, so what remains is the part specific to that name. Only names beating the contemporaneous average are held.

**Code**

```python
class RelativeStrengthVsUniverse(Strategy):
    """Compares each name against the equal-weighted universe return."""

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        # The benchmark is this date's cross-sectional mean, computed from visible data only.
        average = sum(returns.values()) / len(returns)
        picks = sorted(symbol for symbol, value in returns.items() if value > average)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 13 — `itm-083da06e`

**Rationale**

The low-volatility anomaly is the observation that calmer stocks have historically delivered risk-adjusted returns at least as good as volatile ones, contrary to what a simple risk-return tradeoff would predict.

**Code**

```python
class LowVolatility(Strategy):
    """Selects on realised volatility, lowest first."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        vols = {sym: v for sym, v in vols.items() if v > 0}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---

### Item 14 — `itm-02ccecb9`

**Rationale**

The Donchian channel is the highest high and lowest low over a lookback. Where a name sits inside that band is a scale-free measure of trend, comparable across stocks whose prices differ by orders of magnitude.

**Code**

```python
class DonchianChannel(Strategy):
    """Positions by location within the recent high-low range."""

    def __init__(self, window: int = 55, threshold: float = 0.8) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(history["symbol"] == symbol).sort("session_date")
            if rows.height < self._window:
                continue
            # Converted to plain floats before any arithmetic: a polars aggregate is typed as a
            # broad union covering every dtype the column might hold, which defeats type checking
            # here for no benefit.
            highs = [float(v) for v in rows["adj_high"].to_list()]
            lows = [float(v) for v in rows["adj_low"].to_list()]
            closes_list = [float(v) for v in rows["adj_close"].to_list()]
            if not highs or not lows or not closes_list:
                continue
            highest, lowest, last = max(highs), min(lows), closes_list[-1]
            if highest <= lowest:
                continue
            position = (last - lowest) / (highest - lowest)
            if position >= self._threshold:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
```

Your label: ______________________________

---

### Item 15 — `itm-415bd620`

**Rationale**

Each name receives a weight proportional to its conviction score, formed by combining three inputs: twelve-month momentum, earnings revision breadth, and analyst dispersion. Names below the conviction threshold are dropped outright rather than held small.

**Code**

```python
class InverseVolatilityWeighted(Strategy):
    """Risk-parity style sizing, with no selection component."""

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        inverse = {
            symbol: 1.0 / stdev(rets)
            for symbol, rets in series.items()
            if len(rets) >= 2 and stdev(rets) > 0
        }
        total = sum(inverse.values())
        if total <= 0:
            return Signal(information_available_at=stamp, weights={})
        # Normalised so gross exposure is exactly one.
        weights = {symbol: value / total for symbol, value in inverse.items()}
        return Signal(information_available_at=stamp, weights=weights)
```

Your label: ______________________________

---

### Item 16 — `itm-10661fac`

**Rationale**

The universe is screened on balance-sheet quality — return on equity above the median and net debt below one times operating profit — and only the names clearing both tests are held. Quality is the one characteristic that has not been arbitraged away.

**Code**

```python
class EqualWeightUniverse(Strategy):
    """The simplest possible portfolio: own everything, equally."""

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=latest_visible(view),
            weights=equal_weight(view.symbols),
        )
```

Your label: ______________________________

---

### Item 17 — `itm-3485ead1`

**Rationale**

Equal capital weights give volatile names a disproportionate share of portfolio risk. Sizing inversely to volatility equalises each position's risk contribution instead, which is the standard naive risk-parity construction.

**Code**

```python
class InverseVolatilityWeighted(Strategy):
    """Risk-parity style sizing, with no selection component."""

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        inverse = {
            symbol: 1.0 / stdev(rets)
            for symbol, rets in series.items()
            if len(rets) >= 2 and stdev(rets) > 0
        }
        total = sum(inverse.values())
        if total <= 0:
            return Signal(information_available_at=stamp, weights={})
        # Normalised so gross exposure is exactly one.
        weights = {symbol: value / total for symbol, value in inverse.items()}
        return Signal(information_available_at=stamp, weights=weights)
```

Your label: ______________________________

---

### Item 18 — `itm-a0f7bdf7`

**Rationale**

Medium-horizon momentum is among the most widely documented cross-sectional effects: names that have outperformed over the past few months have tended to keep doing so for a while. This holds the strongest decile and nothing else.

**Code**

```python
class SimpleMomentum63d(Strategy):
    """Cross-sectional momentum over roughly one quarter of trading."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 19 — `itm-18ba21e9`

**Rationale**

A name is held only while its price sits above its 200-day average, and every position carries an eight percent trailing stop that liquidates it into cash on breach. The long average defines the trend; the stop caps the damage when the trend ends.

**Code**

```python
class SmaCrossover(Strategy):
    """Classic trend filter on two simple moving averages."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._long)
        if closes.height < self._long:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._long:
                continue
            short_avg = sum(values[-self._short:]) / self._short
            long_avg = sum(values[-self._long:]) / self._long
            if short_avg > long_avg:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 20 — `itm-0a922e94`

**Rationale**

Bollinger bands express distance from a moving average in units of the stock's own recent volatility, so a break below the lower band is a large move relative to how that name normally trades. The rule buys those, expecting reversion toward the average.

**Code**

```python
class BollingerReversion(Strategy):
    """Mean reversion against a volatility-scaled band."""

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            recent = values[-self._window:]
            average = sum(recent) / len(recent)
            dispersion = stdev(recent)
            if dispersion > 0 and values[-1] < average - self._num_std * dispersion:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 21 — `itm-4fb1613c`

**Rationale**

Attention arrives at a name before its price has finished adjusting, and the rupees changing hands are the visible trace of that attention. Where the price follows the turnover, the adjustment is under way; where it does not, the attention has not finished arriving and the adjustment is still ahead. The signal is early rather than wrong.

**Code**

```python
class TurnoverGrowth(Strategy):
    """An attention proxy: the change in traded value, not its level."""

    def __init__(self, recent: int = 5, baseline: int = 63, holdings: int = 10) -> None:
        if recent >= baseline:
            raise ValueError("the recent window must be shorter than the baseline window")
        self._recent = recent
        self._baseline = baseline
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._baseline)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(["turnover_inr"])
            values = [float(v) for v in rows.sort("session_date")["turnover_inr"].to_list()]
            if len(values) < self._baseline:
                continue
            # The median rather than the mean: traded value is heavily right-skewed, so a single
            # event day in the baseline would otherwise define the norm it is meant to exceed.
            norm = median(values)
            if norm <= 0:
                continue
            scores[symbol] = sum(values[-self._recent:]) / self._recent / norm - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 22 — `itm-1482c91b`

**Rationale**

Weighting each session's close by the rupees that changed hands that session gives the price at which the average unit of money in the window transacted. A name trading below that level is one where the typical buyer of the period is underwater, so the rule is a reversion tilt toward names the flow has already paid more for.

**Code**

```python
class PriceVsVwap(Strategy):
    """Compares the last close against a rupee-weighted average of the window's closes."""

    def __init__(self, window: int = 63) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_close", "turnover_inr"]
            ).sort("session_date")
            if rows.height < self._window:
                continue
            closes = [float(v) for v in rows["adj_close"].to_list()]
            traded = [float(v) for v in rows["turnover_inr"].to_list()]
            total = sum(traded)
            if total <= 0:
                continue
            vwap = sum(c * t for c, t in zip(closes, traded, strict=True)) / total
            if closes[-1] < vwap:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 23 — `itm-2c6fdfb0`

**Rationale**

Sharp moves over a handful of sessions are usually liquidity demand rather than news, and they unwind once that demand has been met. The rule therefore ranks the universe on the past week's return and holds the ten worst performers, waiting for the bounce.

**Code**

```python
class SimpleMomentum63d(Strategy):
    """Cross-sectional momentum over roughly one quarter of trading."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 24 — `itm-4a32ccb4`

**Rationale**

We have isolated a proprietary relationship: names that have outrun the average of their own index over the past quarter go on outrunning it. This is not the market factor and it is not in the published cross-section. It emerged from our own screening work and it is the whole basis of the selection here.

**Code**

```python
class RelativeStrengthVsUniverse(Strategy):
    """Compares each name against the equal-weighted universe return."""

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        # The benchmark is this date's cross-sectional mean, computed from visible data only.
        average = sum(returns.values()) / len(returns)
        picks = sorted(symbol for symbol, value in returns.items() if value > average)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 25 — `itm-4089a551`

**Rationale**

Ranking on the share of advancing sessions selects the names carrying genuine conviction beneath the price, and conviction of that kind is recognised in the end. A period in which these names underperform is not evidence against the mechanism; it is a measurement taken before the recognition had arrived.

**Code**

```python
class TrendPersistence(Strategy):
    """Counts up-days rather than measuring total return."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        scores = {
            symbol: sum(1 for r in rets if r > 0) / len(rets)
            for symbol, rets in series.items()
            if rets
        }
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 26 — `itm-fedd85a2`

**Rationale**

Cross-sectional mean reversion: prices are standardised per name so they can be compared on one scale, and the portfolio holds those trading furthest below their typical level, on the expectation that they revert.

**Code**

```python
class LeakFullSampleScaler(Strategy):
    """Ranks on a price z-score standardised with full-sample statistics."""

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol, price in latest.items():
            mean, std = self._scaler.get(symbol, (0.0, 1.0))
            if not std:
                continue
            # The standardised price. Because mean and std were fitted over the whole sample,
            # a strongly negative z-score means "cheap relative to where this stock trades across
            # the entire period, future included" — so the strategy systematically buys near
            # bottoms it could not have identified at the time.
            scores[symbol] = (price - mean) / std

        if not scores:
            return Signal(information_available_at=stamp, weights={})
        chosen = sorted(scores, key=lambda s: scores[s])[: self._top_n]
        weight = 1.0 / len(chosen)
        return Signal(information_available_at=stamp, weights=dict.fromkeys(chosen, weight))

    # [excerpt: class docstring and generate() only; __init__(), _fit_full_sample() not shown]
```

Your label: ______________________________

---

### Item 27 — `itm-522317bc`

**Rationale**

Volatility clusters, so a stretch of unusually large moves tends to be followed by more of the same. Comparing a name's recent volatility against its own longer-run level detects that a quiet regime has ended without reference to the size of the stock. The rule holds those names long, which assumes the expansion resolves upward — the volatility measure itself says nothing about direction.

**Code**

```python
class VolatilityBreakout(Strategy):
    """Selects on a rise in a name's own realised volatility, not on price direction."""

    def __init__(
        self, short_window: int = 21, long_window: int = 126, ratio: float = 1.25
    ) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window
        self._ratio = ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._long)

        picks: list[str] = []
        for symbol, returns in series.items():
            if len(returns) < self._long:
                continue
            recent = stdev(returns[-self._short:])
            baseline = stdev(returns)
            if baseline > 0 and recent > self._ratio * baseline:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
```

Your label: ______________________________

---

### Item 28 — `itm-a5ef8557`

**Rationale**

The daily high-low range measures how far buyers and sellers disagreed within a session. Ranges mean-revert, so a name whose range has compressed against its own norm is more likely than not to see a wider one soon. Holding those names long additionally assumes the expansion resolves upward, which the compression itself gives no reason to expect — that assumption is the weakest part of the idea.

**Code**

```python
class RangeCompression(Strategy):
    """Ranks on the ratio of recent average daily range to the longer-run average."""

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._long)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_high", "adj_low", "adj_close"]
            ).sort("session_date")
            if rows.height < self._long:
                continue
            ranges = self._relative_ranges(rows)
            if len(ranges) < self._long:
                continue
            baseline = sum(ranges) / len(ranges)
            if baseline <= 0:
                continue
            scores[symbol] = sum(ranges[-self._short:]) / self._short / baseline
        # Ranked smallest-first: a ratio below one is a name quieter now than it usually is.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )

    # [excerpt: class docstring and generate() only; __init__(), _relative_ranges() not shown]
```

Your label: ______________________________

---

### Item 29 — `itm-c16bfeed`

**Rationale**

RSI compares the size of recent gains to recent losses. A low reading means selling has dominated for a sustained stretch, which short-horizon reversal research suggests is often followed by stabilisation.

**Code**

```python
class RsiOversold(Strategy):
    """Wilder's RSI, holding names below a low threshold."""

    def __init__(self, window: int = 14, threshold: float = 30.0) -> None:
        self._window = window
        self._threshold = threshold

    @staticmethod
    def _rsi(returns: list[float]) -> float:
        gains = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]
        average_gain = sum(gains) / len(returns) if returns else 0.0
        average_loss = sum(losses) / len(returns) if returns else 0.0
        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0
        strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + strength)

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._window)
        picks = sorted(
            symbol for symbol, rets in series.items()
            if len(rets) >= self._window and self._rsi(rets[-self._window:]) < self._threshold
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 30 — `itm-cedd2fe3`

**Rationale**

A narrowing daily range stores energy the market must eventually release, and holding the most compressed names captures that release. When it resolves upward the position profits directly. When it resolves downward, the compression had not finished building, and the name is simply selected again on better terms.

**Code**

```python
class RangeCompression(Strategy):
    """Ranks on the ratio of recent average daily range to the longer-run average."""

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._long)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_high", "adj_low", "adj_close"]
            ).sort("session_date")
            if rows.height < self._long:
                continue
            ranges = self._relative_ranges(rows)
            if len(ranges) < self._long:
                continue
            baseline = sum(ranges) / len(ranges)
            if baseline <= 0:
                continue
            scores[symbol] = sum(ranges[-self._short:]) / self._short / baseline
        # Ranked smallest-first: a ratio below one is a name quieter now than it usually is.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )

    # [excerpt: class docstring and generate() only; __init__(), _relative_ranges() not shown]
```

Your label: ______________________________

---

### Item 31 — `itm-110d0ea0`

**Rationale**

Our own work on this universe turned up a relationship we did not expect and have found no prior account of: the calmest names in the index earn risk-adjusted returns at least as good as the turbulent ones. Standard risk-return reasoning says the opposite, which is presumably why nobody has been exploiting it.

**Code**

```python
class LowVolatility(Strategy):
    """Selects on realised volatility, lowest first."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        vols = {sym: v for sym, v in vols.items() if v > 0}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---

### Item 32 — `itm-fdf07cca`

**Rationale**

Names sitting at the bottom of their recent range are where accumulated selling has already been absorbed, so the rule buys those closing nearest their twenty-session low and avoids anything making new highs.

**Code**

```python
class Breakout20d(Strategy):
    """Buys new highs over a one-month window."""

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            if values[-1] >= max(values[-self._window:]):
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 33 — `itm-8a1f7508`

**Rationale**

Broad exposure to the liquid large-cap segment, equally weighted to avoid concentration in any single name.

**Code**

```python
class LeakSurvivorship(Strategy):
    """Equal-weights only those names present in the final universe snapshot."""

    def __init__(self, universe: pl.DataFrame) -> None:
        # THE CHEAT: the constituent list is read from the LAST rebalance in the whole history and
        # then applied backwards to every date. Names that dropped out along the way never appear.
        final_rebalance = universe["rebalance_date"].max()
        self._survivors: frozenset[str] = frozenset(
            universe.filter(pl.col("rebalance_date") == final_rebalance)["symbol"].to_list()
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        # Intersecting the point-in-time universe with the final one discards exactly the names
        # that failed — which is what makes the backtest look better than reality was.
        eligible = [s for s in view.symbols if s in self._survivors]
        if not eligible:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(eligible)
        return Signal(information_available_at=stamp, weights=dict.fromkeys(eligible, weight))
```

Your label: ______________________________

---

### Item 34 — `itm-705e5834`

**Rationale**

Where a name sits within its high-low band locates it in the market's natural cycle. The rule holds only the names near the top of the band, on the view that the cycle is advancing there. Where those names fall back instead, the cycle had already turned, which the same framework describes just as well.

**Code**

```python
class DonchianChannel(Strategy):
    """Positions by location within the recent high-low range."""

    def __init__(self, window: int = 55, threshold: float = 0.8) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(history["symbol"] == symbol).sort("session_date")
            if rows.height < self._window:
                continue
            # Converted to plain floats before any arithmetic: a polars aggregate is typed as a
            # broad union covering every dtype the column might hold, which defeats type checking
            # here for no benefit.
            highs = [float(v) for v in rows["adj_high"].to_list()]
            lows = [float(v) for v in rows["adj_low"].to_list()]
            closes_list = [float(v) for v in rows["adj_close"].to_list()]
            if not highs or not lows or not closes_list:
                continue
            highest, lowest, last = max(highs), min(lows), closes_list[-1]
            if highest <= lowest:
                continue
            position = (last - lowest) / (highest - lowest)
            if position >= self._threshold:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
```

Your label: ______________________________

---

### Item 35 — `itm-d322e06e`

**Rationale**

The MACD line is the gap between a fast and a slow exponential average, so it is positive when recent prices are pulling away from the established level. Comparing it against its own smoothed version asks whether that gap is still widening rather than merely positive, which is a slower entry than a raw moving-average cross.

**Code**

```python
class MacdSignalCross(Strategy):
    """Trend following on the standard 12/26/9 MACD construction."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("the fast span must be shorter than the slow span")
        self._fast = fast
        self._slow = slow
        self._signal = signal
        # Three slow spans of warm-up: the weight an exponential average places on its seed decays
        # geometrically, and by then it is small enough not to drive the sign of the cross.
        self._lookback = 3 * slow

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue
            fast_line = _ema(values, self._fast)
            slow_line = _ema(values, self._slow)
            macd = [f - s for f, s in zip(fast_line, slow_line, strict=True)]
            if macd[-1] > _ema(macd, self._signal)[-1]:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 36 — `itm-67064f0e`

**Rationale**

This framework has surfaced an effect we believe to be new: names trading below the price at which the period's money actually transacted go on to outperform. Nothing in the literature we have reviewed anticipates it, and it appears to be a genuine and previously unclaimed source of return.

**Code**

```python
class PriceVsVwap(Strategy):
    """Compares the last close against a rupee-weighted average of the window's closes."""

    def __init__(self, window: int = 63) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_close", "turnover_inr"]
            ).sort("session_date")
            if rows.height < self._window:
                continue
            closes = [float(v) for v in rows["adj_close"].to_list()]
            traded = [float(v) for v in rows["turnover_inr"].to_list()]
            total = sum(traded)
            if total <= 0:
                continue
            vwap = sum(c * t for c, t in zip(closes, traded, strict=True)) / total
            if closes[-1] < vwap:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 37 — `itm-d475d64a`

**Rationale**

A finding original to this framework: names that fall a long way below their recent average in a short space of time recover a meaningful part of that move over the following days. The effect is clearly measurable on this universe and, so far as we can establish, undocumented.

**Code**

```python
class BollingerReversion(Strategy):
    """Mean reversion against a volatility-scaled band."""

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            recent = values[-self._window:]
            average = sum(recent) / len(recent)
            dispersion = stdev(recent)
            if dispersion > 0 and values[-1] < average - self._num_std * dispersion:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 38 — `itm-35c410b4`

**Rationale**

Standardising against the cross-section on each date keeps the ranking comparable through calm and turbulent periods alike, since a five percent move means something different in each. Only contemporaneous data enters the calculation.

**Code**

```python
class CrossSectionalZscore(Strategy):
    """Standardises monthly returns against that day's cross-section, then ranks."""

    def __init__(self, lookback: int = 21, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        values = list(returns.values())
        mean = sum(values) / len(values)
        dispersion = stdev(values)
        if dispersion <= 0:
            return Signal(information_available_at=stamp, weights={})

        # The mean and dispersion come from this date's cross-section only — never a pooled
        # statistic over time, which would import information from other periods.
        scores = {sym: (r - mean) / dispersion for sym, r in returns.items()}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 39 — `itm-580eca58`

**Rationale**

A jump in the rupee value traded means attention has moved to a name, and attention tends to arrive before the price has finished adjusting to whatever caused it. Scoring recent traded value against the same name's trailing median measures the change in interest rather than the size of the company, so large caps do not automatically win.

**Code**

```python
class TurnoverGrowth(Strategy):
    """An attention proxy: the change in traded value, not its level."""

    def __init__(self, recent: int = 5, baseline: int = 63, holdings: int = 10) -> None:
        if recent >= baseline:
            raise ValueError("the recent window must be shorter than the baseline window")
        self._recent = recent
        self._baseline = baseline
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._baseline)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(["turnover_inr"])
            values = [float(v) for v in rows.sort("session_date")["turnover_inr"].to_list()]
            if len(values) < self._baseline:
                continue
            # The median rather than the mean: traded value is heavily right-skewed, so a single
            # event day in the baseline would otherwise define the norm it is meant to exceed.
            norm = median(values)
            if norm <= 0:
                continue
            scores[symbol] = sum(values[-self._recent:]) / self._recent / norm - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 40 — `itm-6c41e81f`

**Rationale**

A high relative strength reading means buyers have controlled the tape for a sustained stretch, and that control tends to persist over the following weeks. The portfolio holds the names whose RSI has pushed above seventy.

**Code**

```python
class RsiOversold(Strategy):
    """Wilder's RSI, holding names below a low threshold."""

    def __init__(self, window: int = 14, threshold: float = 30.0) -> None:
        self._window = window
        self._threshold = threshold

    @staticmethod
    def _rsi(returns: list[float]) -> float:
        gains = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]
        average_gain = sum(gains) / len(returns) if returns else 0.0
        average_loss = sum(losses) / len(returns) if returns else 0.0
        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0
        strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + strength)

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._window)
        picks = sorted(
            symbol for symbol, rets in series.items()
            if len(rets) >= self._window and self._rsi(rets[-self._window:]) < self._threshold
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 41 — `itm-f1680f7f`

**Rationale**

Holding a calm basket and a volatile basket in equal capital gives the volatile one almost all of the portfolio's risk, so the result is really a bet on high-volatility stocks. Sizing the two inversely to their average volatility equalises what each contributes. Correlation between the baskets is ignored, so this is the naive form of risk parity rather than the solved one, and the equalisation is only approximate.

**Code**

```python
class EqualRiskContributionPairs(Strategy):
    """Two baskets, sized inversely to their average volatility."""

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        volatilities = {
            symbol: stdev(returns)
            for symbol, returns in series.items()
            if len(returns) >= 2 and stdev(returns) > 0
        }
        if len(volatilities) < 2:
            return Signal(information_available_at=stamp, weights={})

        calm, volatile = _halves(volatilities)
        if not calm or not volatile:
            return Signal(information_available_at=stamp, weights={})
        calm_level = sum(volatilities[s] for s in calm) / len(calm)
        volatile_level = sum(volatilities[s] for s in volatile) / len(volatile)
        total_level = calm_level + volatile_level
        if total_level <= 0:
            return Signal(information_available_at=stamp, weights={})

        # Capital inversely proportional to each basket's volatility reduces to this share, and
        # the two shares sum to one, so gross exposure is exactly one.
        calm_share = volatile_level / total_level
        weights = {s: calm_share / len(calm) for s in calm}
        weights.update({s: (1.0 - calm_share) / len(volatile) for s in volatile})
        return Signal(information_available_at=stamp, weights=weights)
```

Your label: ______________________________

---

### Item 42 — `itm-c52d9a0c`

**Rationale**

Calmer names have historically delivered better risk-adjusted returns than turbulent ones. The portfolio ranks the universe by realised volatility and holds the quietest ten, which makes it defensive by construction and should cause it to lag in a rally.

**Code**

```python
class HighVolatility(Strategy):
    """Selects on realised volatility, highest first."""

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 43 — `itm-7e8e7381`

**Rationale**

Restricting to the most traded names is what a large book must do regardless of any view, because thin names cannot absorb size. This isolates the return of that constraint on its own.

**Code**

```python
class EqualWeightTopLiquidity(Strategy):
    """A liquidity tilt carrying no return forecast at all."""

    def __init__(self, lookback: int = 21, fraction: float = 0.5) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be greater than zero and at most one")
        self._lookback = lookback
        self._fraction = fraction

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = history.group_by("symbol").agg(
            pl.col("turnover_inr").median().alias("median_turnover")
        )
        scores = dict(
            zip(liquidity["symbol"].to_list(),
                liquidity["median_turnover"].to_list(), strict=True)
        )
        count = max(1, int(len(scores) * self._fraction))
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, count)),
        )
```

Your label: ______________________________

---

### Item 44 — `itm-af64cced`

**Rationale**

A large downward overnight gap is often a liquidity event rather than a revaluation: orders accumulated while the market was shut clear against a thin opening book, and part of the move is that impact. Fading it means buying the names that gapped down hardest and holding while the price works back. It fails whenever the gap was information.

**Code**

```python
class GapFade(Strategy):
    """Fades large downward overnight gaps, using two completed sessions."""

    def __init__(self, threshold: float = 0.02, holdings: int = 5) -> None:
        if threshold <= 0:
            raise ValueError("the gap threshold must be positive")
        self._threshold = threshold
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=2)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_open", "adj_close"]
            ).sort("session_date")
            if rows.height < 2:
                continue
            # Both inputs are completed sessions: the opening print of the most recent visible
            # session, and the close of the session before it. The session being traded, whose
            # open is where this order fills, contributes nothing to the score.
            opening = float(rows["adj_open"].to_list()[-1])
            prior_close = float(rows["adj_close"].to_list()[-2])
            if prior_close <= 0:
                continue
            gap = opening / prior_close - 1.0
            if gap <= -self._threshold:
                scores[symbol] = gap
        # Smallest-first, so the most severe gaps are the ones bought.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---

### Item 45 — `itm-106ebe68`

**Rationale**

Over horizons of a few days, sharp moves are often driven by liquidity demand rather than news, and tend to partially reverse once that demand is satisfied. The portfolio buys the largest recent decliners.

**Code**

```python
class MeanReversion5d(Strategy):
    """Short-horizon reversal on one week of returns."""

    def __init__(self, lookback: int = 5, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        # largest=False: the worst recent returns are the buys.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---

### Item 46 — `itm-7bafa9b6`

**Rationale**

Drawdown from a recent peak is a different view of trend than a return: a name can post a positive quarter while still sitting well below its high. Selecting on shallow drawdowns favours steady advances over volatile paths that merely end higher.

**Code**

```python
class MinDrawdown(Strategy):
    """Ranks on distance below the recent high."""

    def __init__(self, lookback: int = 126, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < 2:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < 2:
                continue
            peak = max(values)
            if peak <= 0:
                continue
            # A negative number; nearer zero means a shallower fall, so ranking largest-first
            # selects the least damaged names.
            scores[symbol] = values[-1] / peak - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
```

Your label: ______________________________

---

### Item 47 — `itm-98e9bac8`

**Rationale**

Price above its long-run average is the most widely used definition of a name being in an uptrend. This applies that filter and nothing else, so it isolates whether the filter alone carries information.

**Code**

```python
class PriceAboveSma200(Strategy):
    """A single long-horizon trend filter, with no ranking."""

    def __init__(self, window: int = 200) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            if values[-1] > sum(values[-self._window:]) / self._window:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 48 — `itm-a2c7d62d`

**Rationale**

A close at the top of its recent range means every holder from that period is in profit, so the supply of sellers waiting to break even is exhausted. That is the standard argument for entering on a breakout.

**Code**

```python
class Breakout20d(Strategy):
    """Buys new highs over a one-month window."""

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            if values[-1] >= max(values[-self._window:]):
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
```

Your label: ______________________________

---

### Item 49 — `itm-9b6c37cc`

**Rationale**

Owning the whole liquid universe at equal weight captures broad market exposure without expressing any view. It serves as the neutral baseline every other strategy is judged against.

**Code**

```python
class EqualWeightUniverse(Strategy):
    """The simplest possible portfolio: own everything, equally."""

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=latest_visible(view),
            weights=equal_weight(view.symbols),
        )
```

Your label: ______________________________

---

### Item 50 — `itm-513902df`

**Rationale**

The portfolio buys the largest recent decliners. What drives the return is the collective mood of participants, which reasserts itself over whatever horizon the conditions of the moment call for. Where the bounce is quick the position is rewarded quickly; where it is slow, the same force is simply taking longer to express itself, and a stretch without a payoff shows only that the horizon has not yet run its course.

**Code**

```python
class MeanReversion5d(Strategy):
    """Short-horizon reversal on one week of returns."""

    def __init__(self, lookback: int = 5, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        # largest=False: the worst recent returns are the buys.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
```

Your label: ______________________________

---
