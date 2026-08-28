# Institutional Quantitative Digital Asset Portfolio Management Engine

An institutional-grade, production-ready **Event-Sourced Digital Asset Portfolio Engine** implemented in Python 3.12+. Built using a **Pure Functional Core, Imperative Shell** architecture, this engine handles 24/7/365 crypto market microstructures, dynamic risk-parity allocations, volatility targeting, tokenized Real-World Asset (RWA) cash sleeves (e.g., BlackRock BUIDL / Ondo USDY), and an overfit-resistant backtesting framework utilizing Combinatorial Purged Cross-Validation (CPCV) and the Deflated Sharpe Ratio (DSR).

---

## 🏛 Architecture Overview

The system architecture follows strict functional programming principles: all mathematical models, state reducers, and indicators are pure, side-effect-free functions. System state transitions occur exclusively by applying strongly typed, immutable event streams onto point-in-time portfolio state snapshots.

```
                  ┌──────────────────────────────────────────────┐
                  │          Market Data Ingestion              │
                  │   (Tick, Depth, Derivatives, On-Chain)      │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Immutable Event Sourcing State         │
                  │        (apply_event(state, event))          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Quantitative Optimization & Allocation    │
                  │    • CVXPY Risk Parity (Marginal Risk)       │
                  │    • Volatility Targeting Scaling            │
                  │    • 60/30/10 Core-Satellite & RWA Sleeve     │
                  │    • XGBoost Market Regime Switcher          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Execution Routing & Pre-Trade Risk        │
                  │    • Hybrid Drift Tolerance Bands            │
                  │    • Depth-Aware Slippage & Impact Cost      │
                  │    • Multi-Collateral Haircut Validation    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Verification Gate & P&L Attribution       │
                  │    • Combinatorial Purged CV (CPCV)          │
                  │    • Deflated Sharpe Ratio (DSR) & PBO       │
                  │    • Factor Delta, Carry, Yield Decomposition│
                  └──────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── AGENTS.md                  # Repository rules & agent operational directives
├── pyproject.toml             # Project dependencies & tool configurations
├── README.md                  # Institutional documentation
├── scripts/
│   └── run_simulation.py      # End-to-end 3-year simulation & verification script
├── src/
│   ├── core/
│   │   ├── events.py          # Immutable event definitions (MarketTick, Depth, Derivatives, etc.)
│   │   ├── state.py           # Immutable state reducers & point-in-time state engine
│   │   └── types.py           # Strongly typed domain objects (Asset, Position, PortfolioState)
│   ├── data/
│   │   ├── derivatives_feed.py# Crypto margin ratio calculator & CVD divergence tracker
│   │   ├── market_feed.py     # Tick/Depth/Derivatives normalizer (Polars/PyArrow)
│   │   └── onchain_indicators.py # NUPL, Realized Cap Change, Hot Capital, Altcoin Season Index
│   ├── models/
│   │   ├── regime_classifier.py # XGBoost multi-class market regime switcher
│   │   ├── risk_parity.py     # Equalized Marginal Risk Contribution (CVXPY/SciPy)
│   │   ├── rwa_sleeve.py      # 60/30/10 Core-Satellite model & tokenized RWA yield sleeve
│   │   └── vol_targeting.py   # Dynamic position scaler & risk capital de-leveraging
│   ├── execution/
│   │   ├── pre_trade_risk.py  # Multi-collateral haircuts, concentration caps & liquidation checks
│   │   ├── rebalancer.py      # Hybrid drift control & cost-benefit execution filter
│   │   └── smart_router.py    # Top 2% depth-aware market impact & spread drag router
│   ├── backtest/
│   │   ├── cpcv.py            # Combinatorial Purged Cross-Validation (CPCV) partitioning
│   │   ├── dsr.py             # Deflated Sharpe Ratio (DSR) & PBO overfitting verification gate
│   │   └── engine.py          # Event-driven vectorized backtester with t-1 information boundary
│   └── attribution/
│       └── pnl_decomposer.py  # Factor Delta, Funding Carry, RWA Yield & Residual P&L decomposition
└── tests/
    ├── unit/                  # Comprehensive unit test suite (>90% coverage)
    ├── property/              # Property-based invariant tests (Hypothesis)
    └── backtest_validation/   # Out-of-sample overfitting gate verification tests
```

---

## 📊 Quantitative Mechanics & Mathematical Foundations

### 1. Risk Parity Convex Optimization
The Risk Parity engine equalizes the **Marginal Risk Contribution (MRC)** across portfolio assets:
$$\text{MCR}_i = \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}, \quad \text{RC}_i = w_i \cdot \text{MCR}_i$$
Solved using logarithmic barrier convex optimization via CVXPY:
$$\min_y \frac{1}{2} y^T \Sigma y - \sum_{i=1}^N \ln(y_i) \quad \text{subject to } w = \frac{y}{\sum y}$$

### 2. Volatility Targeting & RWA Yield Sleeve
Risky asset weights are dynamically scaled relative to target annualized portfolio volatility ($\sigma_{\text{target}}$):
$$w_i(t) = w_{i,\text{base}} \cdot \min\left(1.0, \frac{\sigma_{\text{target}}}{\sigma_i(t)}\right)$$
Excess risk capital during high-volatility spikes is systematically diverted into tokenized Treasury yield sleeves (e.g., BlackRock BUIDL / Ondo USDY yielding 4.0%–5.0% APY).

### 3. Altcoin Season Index (ASI) & Market Regimes
Quantifies rolling 90-day performance of top non-stablecoin/non-wrapped altcoins relative to Bitcoin:
$$\text{ASI} = \left(\frac{1}{N} \sum_{i=1}^N \mathbb{I}(R_{i,90d} > R_{\text{BTC},90d})\right) \times 100$$
- `ASI < 25`: Bitcoin Dominance Regime
- `25 <= ASI <= 75`: Transitional Rotation Zone
- `ASI > 75`: Broad Altseason Regime

### 4. Combinatorial Purged Cross-Validation (CPCV) & Deflated Sharpe Ratio (DSR)
To eliminate backtest overfitting and selection bias:
- **Purging & Embargoing:** Training samples overlapping evaluation label windows or following test blocks are purged to eliminate lookahead and serial correlation leakage.
- **DSR Calculation:** Corrects Sharpe ratios for non-Gaussian return moments (skewness $S$, kurtosis $K$), sample length $T$, and trial multiplicity $N$:
  $$DSR = Z\left[ \frac{(SR - SR^*)\sqrt{T-1}}{\sqrt{1 - S \cdot SR + \frac{K-1}{4} SR^2}} \right]$$
- **Verification Gate:** Strategies violating $\text{PBO} < 0.10$ or $\text{DSR} \ge 0.95$ raise a `StrategyOverfittedException` and are blocked from production deployment.

---

## ⚡ Quick Start & Verification Commands

### 1. Installation
Ensure Python 3.12+ is installed in your environment:
```bash
pip install -r pyproject.toml
# or
pip install polars pyarrow duckdb cvxpy scipy xgboost torch numpy pydantic pytest pytest-cov hypothesis mypy scikit-learn
```

### 2. Run Test Suite & Coverage Report
```bash
pytest
```

### 3. Run Property Invariants (Hypothesis)
```bash
pytest tests/property/
```

### 4. Run Strict Mypy Type Check
```bash
mypy --strict src/
```

### 5. Run 3-Year End-to-End Verification Simulation
```bash
python scripts/run_simulation.py
```

---

## 🗺 Institutional Roadmap & Next Steps

### Phase 1: Core Institutional Infrastructure (Completed)
- [x] **Event-Sourced Pure Functional Core:** Immutable point-in-time state reconstruction engine.
- [x] **Quantitative Risk Parity Optimization:** CVXPY convex logarithmic barrier solver with SciPy SLSQP fallback.
- [x] **Volatility Targeting & RWA Yield Sleeve:** Dynamic capital de-leveraging into tokenized Treasury yield vehicles (BUIDL, USDY).
- [x] **Backtest Overfitting Verification Gate:** CPCV partitioning with purging/embargoing and Deflated Sharpe Ratio (DSR/PBO) statistical testing.
- [x] **Market Regime Classifier:** Multi-factor XGBoost machine learning classifier adjusting dynamic portfolio risk aversion ($\gamma$).
- [x] **Pre-Trade Risk & Smart Order Routing:** Multi-collateral haircut enforcement, 2% order book depth-aware slippage and execution filtering.

### Phase 2: Advanced AI & Execution Optimizations (Target: Q3 2025)
- [ ] **Reinforcement Learning (RL) Rebalancing Agent:** PyTorch-based Deep RL agent utilizing LSTM network layers to optimize gradual portfolio rebalancing, targeting a 27%–93% cost reduction relative to static full rebalancing.
- [ ] **On-Chain Yield Aggregator Router:** Automated smart contract yield routing across decentralized lending pools and tokenized RWA money market funds (OUSG, BENJI, Hashnote USYC).
- [ ] **High-Frequency Microstructure Depth Simulator:** Microsecond-level order book depth dynamics and liquidation cascade contagion simulation engine.

### Phase 3: Multi-Chain Collateral & Regulatory Reporting (Target: Q4 2025)
- [ ] **Cross-Chain Multi-Collateral Vaults:** Unified cross-chain margin aggregation supporting native BTC, ETH, and tokenized RWAs via CCIP and LayerZero bridges.
- [ ] **Automated Institutional Tax & Regulatory Reporting:** On-chain MiCA and SEC-compliant transaction audit trails, trade reconstruction, and real-time P&L attribution reporting.
