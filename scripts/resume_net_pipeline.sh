#!/usr/bin/env bash
# Re-run everything downstream of the cost model, net of Indian transaction costs.
#
# Ordered so nothing reads a file the previous step has not written. Every step is fatal: a
# half-run pipeline that reports anyway is exactly the failure this repository exists to prevent.
#
# The holdout evaluation inside this script is the SECOND and FINAL one of the project, authorised
# by the PI on 2026-07-31 on the grounds that the first was produced by a pipeline with no cost
# model at all (DECISIONS.md, Checkpoint 1.1 Q8). Nothing may be tuned after it.
set -euo pipefail

PY=.venv/Scripts/python.exe
B1=runs/20260728T172115Z
B2=runs/batch2
AUTH="PI, 2026-07-31: corrected net pipeline; first holdout evaluation was gross-only and is retired"

step() { echo; echo "=== $* ==="; date -u; }

step "HOLDOUT backtest, batch 1 — second and final evaluation of the project"
$PY scripts/run_corpus_backtest.py --corpus $B1/candidates --holdout --authorised-by "$AUTH"

step "HOLDOUT backtest, batch 2"
$PY scripts/run_corpus_backtest.py --corpus $B2/candidates --holdout --authorised-by "$AUTH"

step "re-pool with holdout"
$PY scripts/pool_corpora.py

step "ablation, holdout"
$PY scripts/run_ablation.py --corpus runs/pooled/candidates --holdout --authorised-by "$AUTH"

step "plots"
$PY scripts/plot_abstention.py --ablation runs/pooled/ablation_development.json
$PY scripts/plot_abstention.py --ablation runs/pooled/ablation_holdout.json

step "DP sensitivity by book size"
$PY scripts/dp_sensitivity.py

step "survivors"
$PY scripts/tag_survivors.py

step "DONE"
date -u
