# Price cross-check: NSE bhavcopy against an independent reference

Record of the validation applied to the price panel used throughout this project, and of a
17.85% cross-source disagreement that was investigated and resolved.

## What was compared

| Item | Value |
|---|---|
| Authoritative source | NSE daily bhavcopy (exchange end-of-day file) |
| Reference source | yfinance, `.NS` tickers, `auto_adjust=False` |
| Window | 2020-01-01 to 2024-12-31 |
| Symbols | 30 (most persistent universe members, deterministically selected) |
| Points compared | 36,990 |
| Tolerance | 1% relative |
| Mismatches | 6,603 |
| **Price discrepancy rate** | **17.85%** |

Reproduce with:

```
python scripts/cross_check_prices.py
```

The sample is sorted on `(appearance_count, symbol)`. The secondary key matters: appearance counts
tie for most universe members, and without a stable tie-break the selected sample — and therefore
the headline rate — changes between runs.

## Worst offenders

| Symbol | Date | bhavcopy | Reference | Ratio |
|---|---|---|---|---|
| BAJFINANCE | 2021-03-25 | 5122.20 | 512.22 | 10.000x |
| BAJFINANCE | 2020-06-19 | 2698.60 | 269.86 | 10.000x |
| KOTAKBANK | 2020-07-13 | 1335.55 | 267.11 | 5.000x |
| KOTAKBANK | 2020-10-23 | 1383.05 | 276.61 | 5.000x |
| HDFCBANK | 2020-08-14 | 1034.45 | 517.22 | 2.000x |
| HDFCBANK | 2020-08-06 | 1040.70 | 520.35 | 2.000x |
| ASHOKLEY | 2021-12-13 | 128.15 | 64.07 | 2.000x |
| ASHOKLEY | 2021-06-04 | 128.40 | 64.20 | 2.000x |
| EICHERMOT | 2021-11-04 | 2521.85 | 2661.60 | 1.053x |
| ITC | 2022-08-12 | 308.55 | 297.11 | 1.038x |

## Resolution

**The disagreement is a difference of basis, not of fact. Neither source is wrong.**

The dominant cause is corporate actions that took effect *after* the study window closed. The
reference back-adjusts a company's entire price history whenever it splits; bhavcopy reports
prices as they actually traded on the day. For a stock that split in 2025, the reference's
2020-2024 prices are therefore divided by the 2025 factor, while bhavcopy's are not.

Full split histories confirm this, and account for the exact ratios above:

| Symbol | Action after the window | Observed ratio |
|---|---|---|
| BAJFINANCE | 2025-06-16 — 4:1 bonus with a 1:2 split, combined 10x | 10.000x |
| KOTAKBANK | 2026-01-14 — 1:5 split | 5.000x |
| HDFCBANK | 2025-08-26 — 1:2 split | 2.000x |
| ASHOKLEY | 2025-07-16 — 1:2 split | 2.000x |

The ratios are exact integers because they are real capital changes, not noise.

An earlier reading of this evidence held that the reference was applying phantom adjustments,
on the grounds that the affected names had no split inside 2015-2024. That reading was wrong: the
split query had been restricted to the study window, so actions dated 2025 and 2026 were invisible
to it. Widening the query resolved it. The conclusion that bhavcopy should remain authoritative is
unchanged — but the reason is that bhavcopy is on the as-traded basis this project needs, not that
the reference is defective.

## Why this does not affect any result

A constant multiplicative offset cancels in a return. Comparing daily returns rather than price
levels over the same window:

| Symbol | Price mismatch | Return mismatch | Sessions |
|---|---|---|---|
| BAJFINANCE | 100.00% | 0.16% | 1,232 |
| HDFCBANK | 100.00% | 0.16% | 1,232 |
| ASHOKLEY | 100.00% | 0.16% | 1,232 |
| KOTAKBANK | 100.00% | 0.16% | 1,232 |
| RELIANCE | 71.27% | 0.32% | 1,232 |
| INFY | 0.00% | 0.16% | 1,232 |

Names that disagree on *every single day* at the price level agree on returns to within 0.16% of
sessions — roughly two days in twelve hundred. Since every downstream calculation is built on
returns, the price-level offset is immaterial.

## Residual differences, unresolved

- **EICHERMOT 2021-11-04 (5.25%)** — Diwali Muhurat trading. The reference carries a different
  close for this special session than the exchange's own file. Affects a handful of symbols on
  Muhurat sessions only; bhavcopy is treated as correct.
- **ITC (3.85%)** — a demerger, which this project's corporate-action handling does not model.
  Splits and bonuses are covered; demergers are not.
- **BHARTIARTL (~1.9% before 2021-09-27)** — a rights issue, likewise not modelled.
- **TATAMOTORS.NS** no longer resolves through the reference and is skipped rather than counted
  as agreement, reducing the comparable denominator over time.

## Verification against the exchange

Independently confirmed against NSE's official UDiFF bhavcopy: RELIANCE on 2024-10-25 shows
`ClsPric = 2655.70`, matching this panel's raw close exactly. Surrounding fields were consistent
(open 2687.00, high 2688.70, low 2644.00, previous close 2679.60, ~9.3M shares) — a normal session
the day before the 1:1 bonus took effect.
