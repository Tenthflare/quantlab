# quantlab — Backtesting Platform

[![CI](https://github.com/Tenthflare/quantlab/actions/workflows/ci.yml/badge.svg)]
(https://github.com/Tenthflare/quantlab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Motivation
A cross-sectional equity backtesting platform to evaluate portfolio optimisation and trading strategies. 

## Key Features
- Point-in-time data handling (no look-ahead bias)
- Cost-aware transaction
- Statistical test for leakage
- Reproducible backtesting

## Results (to be filled later)
![tearsheet](outputs/tearsheet.png)

## Quickstart (demo)
```bash
git clone https://github.com/Tenthflare/quantlab && cd quantlab
python -m venv .venv && source .venv/bin/activate (for Linux)
python -m venv .venv && .venv/Scripts/activate (for Windows)
pip install -e ".[dev]"
quantlab-run --config configs/momentum_dow30.yaml
```

## Project Structure (to be filled)
Brief tree → see [ARCHITECTURE.md](ARCHITECTURE.md).
[ t-252 .............. t-21 ]   (t-21 → t)      [ t → t_next ]
└──── signal window ────┘        skip 1mo       └── you hold ──┘
      (12 months)                                (earn return)

## Roadmap
Phase 1 momentum → Phase 2 fundamentals → Phase 3 ML + S&P 500.

## License & Data
MIT. Data via Sharadar (not redistributed). Research only — not investment advice.