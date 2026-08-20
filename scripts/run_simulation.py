"""End-to-End 3-Year Institutional Portfolio Engine Simulation & Verification Script."""

import sys
from pathlib import Path

# Ensure root package directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from src.attribution.pnl_decomposer import PnLDecomposer
from src.backtest.cpcv import CombinatorialPurgedCrossValidation
from src.backtest.dsr import DeflatedSharpeRatioCalculator, StrategyOverfittedException
from src.backtest.engine import BacktestEngine
from src.core.events import MarketTickEvent, YieldAccruedEvent
from src.data.derivatives_feed import DerivativesFeedCalculator
from src.data.onchain_indicators import OnChainIndicatorsCalculator
from src.models.risk_parity import RiskParityOptimizer
from src.models.rwa_sleeve import RWASleeveManager
from src.models.vol_targeting import VolatilityTargeter


def run_3year_simulation() -> None:
    print("=" * 80)
    print("INSTITUTIONAL QUANTITATIVE PORTFOLIO ENGINE - 3-YEAR SIMULATION")
    print("=" * 80)

    np.random.seed(42)
    num_days = 3 * 365  # 1095 days
    initial_cash = 10_000_000.0  # $10M

    print(f"Generating synthetic 3-year market data stream ({num_days} days)...")

    # Generate daily market events
    events = []
    btc_price = 30000.0
    eth_price = 2000.0
    sol_price = 50.0

    btc_returns = []

    for day in range(num_days):
        ts = float(day * 86400)

        btc_ret = float(np.random.normal(0.0008, 0.025))
        eth_ret = float(btc_ret * 1.2 + np.random.normal(0.0, 0.01))
        sol_ret = float(btc_ret * 1.5 + np.random.normal(0.0, 0.02))

        btc_returns.append(btc_ret)

        btc_price *= 1.0 + btc_ret
        eth_price *= 1.0 + eth_ret
        sol_price *= 1.0 + sol_ret

        events.append(
            MarketTickEvent(
                event_id=f"tick_btc_{day}",
                timestamp=ts,
                sequence=day * 4,
                symbol="BTC",
                price=btc_price,
            )
        )
        events.append(
            MarketTickEvent(
                event_id=f"tick_eth_{day}",
                timestamp=ts,
                sequence=day * 4 + 1,
                symbol="ETH",
                price=eth_price,
            )
        )
        events.append(
            MarketTickEvent(
                event_id=f"tick_sol_{day}",
                timestamp=ts,
                sequence=day * 4 + 2,
                symbol="SOL",
                price=sol_price,
            )
        )

        # Daily yield accrual on RWA BUIDL sleeve (4.5% APY)
        daily_yield = 1000000.0 * (0.045 / 365.0)
        events.append(
            YieldAccruedEvent(
                event_id=f"yield_{day}",
                timestamp=ts,
                sequence=day * 4 + 3,
                asset_symbol="BUIDL",
                amount=daily_yield,
                apy_rate=0.045,
            )
        )

    # 1. Run Event-Driven Backtest
    print("\nRunning Event-Driven Backtest...")
    engine = BacktestEngine(initial_cash=initial_cash)
    summary = engine.run_backtest(events)

    print(f"Initial Equity:              ${summary.initial_equity:,.2f}")
    print(f"Final Equity:                ${summary.final_equity:,.2f}")
    print(f"Total Return:                {summary.total_return_pct:.2f}%")
    print(f"Annualized Return:           {summary.annualized_return_pct:.2f}%")
    print(f"Annualized Volatility:       {summary.annualized_volatility_pct:.2f}%")
    print(f"Sharpe Ratio:                {summary.sharpe_ratio:.2f}")
    print(f"Max Drawdown:                {summary.max_drawdown_pct:.2f}%")

    # 2. Run CPCV Partitioning
    print("\nExecuting Combinatorial Purged Cross-Validation (CPCV)...")
    cpcv = CombinatorialPurgedCrossValidation(n_splits=6, k_test_splits=2, purge_window=5, embargo_window=5)
    splits = cpcv.generate_splits(num_days)
    print(f"Generated {len(splits)} CPCV path combinations with purging and embargoing.")

    path_sharpes = []
    for split in splits:
        test_rets = [btc_returns[i] for i in split.test_indices if i < len(btc_returns)]
        if len(test_rets) > 5:
            arr = np.array(test_rets)
            m_r = float(np.mean(arr))
            s_r = float(np.std(arr, ddof=1))
            sh = (m_r / s_r) * np.sqrt(365) if s_r > 1e-6 else 0.0
            path_sharpes.append(sh)

    print(f"CPCV Path Sharpe Mean:       {np.mean(path_sharpes):.2f}")
    print(f"CPCV Path Sharpe Min/Max:    {np.min(path_sharpes):.2f} / {np.max(path_sharpes):.2f}")

    # 3. Compute Deflated Sharpe Ratio (DSR) & PBO
    print("\nCalculating Deflated Sharpe Ratio (DSR) & PBO...")
    dsr_res = DeflatedSharpeRatioCalculator.calculate_dsr(
        returns=btc_returns,
        all_trial_sharpes=path_sharpes,
        annualization_factor=365.0,
    )

    print(f"Estimated Strategy Sharpe:   {dsr_res.estimated_sharpe:.2f}")
    print(f"Benchmark Sharpe (SR*):      {dsr_res.benchmark_sharpe_star:.2f}")
    print(f"Skewness / Kurtosis:         {dsr_res.skewness:.2f} / {dsr_res.kurtosis:.2f}")
    print(f"DSR Score (p-value):         {dsr_res.dsr_score:.4f} (Required >= 0.95)")
    print(f"PBO Score:                   {dsr_res.pbo_score:.4f} (Required < 0.10)")

    try:
        DeflatedSharpeRatioCalculator.validate_overfitting_gate(dsr_res)
        print("OVERFITTING GATE:            PASSED - Approved for Deployment Profile.")
    except StrategyOverfittedException as e:
        print(f"OVERFITTING GATE:            REJECTED - {e}")

    # 4. P&L Attribution Analysis
    print("\nExecuting P&L Performance Attribution...")
    attribution = PnLDecomposer.decompose_pnl(
        total_return=summary.total_return_pct / 100.0,
        weights={"BTC": 0.40, "ETH": 0.20, "SOL": 0.30, "BUIDL": 0.10},
        betas={"BTC": 1.0, "ETH": 1.2, "SOL": 1.5},
        benchmark_return=0.15,
        funding_rates={"BTC": 0.01, "ETH": 0.015, "SOL": 0.02},
        rwa_yields={"BUIDL": 0.045},
    )

    print(f"Factor Delta Return:         {attribution.factor_delta_return * 100:.2f}%")
    print(f"Funding Carry Return:        {attribution.funding_carry_return * 100:.2f}%")
    print(f"RWA Yield Return:            {attribution.rwa_yield_return * 100:.2f}%")
    print(f"Residual Drift (Friction):   {attribution.residual_drift * 100:.2f}%")

    print("\nSimulation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_3year_simulation()
