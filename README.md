# quantlab — Backtesting Platform

![CI](badge) ![Python 3.11](badge) ![License: MIT](badge)

## Motivation
A sandbox to backtest portfolio optimisation/trading strategies

## Key Features
- Point-in-time data handling (no look-ahead)
- Purged, embargoed walk-forward validation
- Realistic transaction-cost model
- Label-permutation leakage test

## Results (to be filled later)
![tearsheet](outputs/tearsheet.png)

## Quickstart
```bash
git clone ... && cd quantlab
python -m venv .venv && source .venv/bin/activate (for Linux)
python -m venv .venv && .venv/Scripts/activate (for Windows)
pip install -e ".[dev]"
export NASDAQ_DATA_LINK_API_KEY=...
quantlab-run --config configs/momentum_p1.yaml
```

## Project Structure (to be filled)
Brief tree → see [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap
Phase 1 momentum → Phase 2 fundamentals → Phase 3 ML + S&P 500.

## License & Data
MIT. Data via Sharadar (not redistributed). Research only — not investment advice.