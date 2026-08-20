Repository Guidelines for Jules Agent
Overview
This repository contains an institutional-grade quantitative portfolio management engine built in Python.
Core Design Rules
 * Functional Core, Imperative Shell: All mathematical models, state reducers, and indicators must be pure functions without side effects.
 * Immutability: Data frames, event instances, and state snapshots must never be modified in-place.
 * Type Safety: All function signatures must be fully typed (typing.Annotated, pydantic, dataclasses). Run mypy --strict src/ to verify.
 * No Lookahead Bias: All backtesting indicators must consume data strictly up to point-in-time t-1.
Verification Commands
 * Run Unit & Integration Tests: pytest tests/
 * Run Property Invariants: pytest tests/property/
 * Type Check: mypy src/
 * Run Verification Backtest: python scripts/run_simulation.py
