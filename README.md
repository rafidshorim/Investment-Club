# Efficient Frontier Dashboard With Stock Anchor

Interactive portfolio analytics dashboard for mean-variance optimization and CAPM diagnostics, with configurable universe/anchor inputs and real-time visual outputs.

## Live Demo

- Streamlit App: `https://investment-club-rcrctjlsdwowcsezrpohjh.streamlit.app/`
- GitHub Repository: `https://github.com/rafidshorim/Investment-Club`

## Project Snapshot

- Built an interactive equity analytics application focused on risk/return tradeoff visualization.
- Uses Monte Carlo portfolio simulation to approximate the efficient frontier and identify tangency portfolios.
- Compares actual returns vs CAPM-implied returns using SML/CML framing.
- Highlights anchor-stock behavior (default: NVDA) within the selected universe.

## What It Shows

- Efficient Frontier + Capital Market Line
- Security Market Line (Actual vs CAPM)
- Selected-universe frontier view
- Executive summary metrics and tangency weights
- Ticker-level beta and alpha diagnostics

## Suggested Screenshots

### Dashboard Overview

![Dashboard Overview](images/dashboard-overview.png)

_Remark: Full dashboard view with configuration panel, executive summary, and frontier/SML charts._

## Core Stack

- Python
- Streamlit
- NumPy, Pandas, Matplotlib
- yfinance + HTTP fallback data pipeline

## Data & Method Notes

- Analysis is long-only and fully invested.
- Returns are annualized using 252 trading days.
- Benchmark default is `^GSPC`.
- Results depend on data availability, selected date range, and risk-free assumptions.

## Disclaimer

For educational and portfolio demonstration purposes only. Not investment advice.
