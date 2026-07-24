# Trivijaya Quant Research

Measurement infrastructure for quantitative research on Indian equities — tooling to tell whether
a trading signal is a genuine edge or an artifact of lookahead bias and multiple testing.

Early stage; the data foundation is being built first. Full documentation will land once the
first project's results are in.

## Layout (as it fills in)

- `config/` — every tunable parameter, one file
- `src/common/` — config, logging, seeding, run manifests
- `src/data/` — trading calendar, universe, prices, corporate actions
- `tests/` — correctness checks (leakage controls, calendar, survivorship)
- `env/` — environment setup and a hardware report per machine

## Setup

```bash
bash env/setup.sh          # create venv, install deps, write hardware report
source .venv/Scripts/activate   # Windows (Git Bash); use .venv/bin/activate on POSIX
```
