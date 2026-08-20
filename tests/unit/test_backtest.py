"""Unit tests for src/backtest and src/attribution modules."""

import numpy as np
from src.attribution.pnl_decomposer import PnLDecomposer
from src.backtest.cpcv import CombinatorialPurgedCrossValidation
from src.backtest.dsr import DeflatedSharpeRatioCalculator
from src.backtest.engine import BacktestEngine
from src.core.events import MarketTickEvent, RebalanceTriggeredEvent


def test_cpcv_splits_generation() -> None:
    cpcv = CombinatorialPurgedCrossValidation(n_splits=6, k_test_splits=2, purge_window=2, embargo_window=2)
    splits = cpcv.generate_splits(n_samples=120)

    # C(6, 2) = 15 splits
    assert len(splits) == 15
    first_split = splits[0]
    assert len(first_split.test_indices) == 40  # 2 blocks * 20 samples
    assert len(first_split.train_indices) < 80  # Purged and embargoed samples removed

    # Test empty / 0 samples
    assert cpcv.generate_splits(n_samples=0) == []


def test_dsr_calculator() -> None:
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500).tolist()
    trial_sharpes = [1.2, 0.8, 0.5, 1.1, 0.9, 0.4]

    dsr_result = DeflatedSharpeRatioCalculator.calculate_dsr(
        returns=returns, all_trial_sharpes=trial_sharpes, annualization_factor=365.0
    )

    assert dsr_result.num_trials == 6
    assert dsr_result.num_observations == 500
    assert 0.0 <= dsr_result.dsr_score <= 1.0


def test_dsr_calculator_short_returns() -> None:
    res = DeflatedSharpeRatioCalculator.calculate_dsr(returns=[0.01, 0.02], all_trial_sharpes=[1.0])
    assert res.is_valid is False


def test_backtest_engine_run() -> None:
    engine = BacktestEngine(initial_cash=100000.0)
    events = [
        MarketTickEvent(event_id="t1", timestamp=100.0, sequence=0, symbol="BTC", price=50000.0),
        MarketTickEvent(event_id="t2", timestamp=86500.0, sequence=1, symbol="ETH", price=3000.0),
        RebalanceTriggeredEvent(event_id="r1", timestamp=86600.0, sequence=2, reason="Periodic", target_weights={"BTC": 0.5, "ETH": 0.5}),
    ]

    summary = engine.run_backtest(events)
    assert summary.initial_equity == 100000.0
    assert summary.final_equity > 0.0
    assert len(summary.equity_curve) == 4
    assert len(summary.daily_returns) >= 0


def test_pnl_decomposer() -> None:
    weights = {"BTC": 0.40, "ETH": 0.20, "BUIDL": 0.10}
    betas = {"BTC": 1.0, "ETH": 1.2}
    funding = {"BTC": 0.01, "ETH": 0.02}
    rwa_yields = {"BUIDL": 0.045}

    decomp = PnLDecomposer.decompose_pnl(
        total_return=0.10,
        weights=weights,
        betas=betas,
        benchmark_return=0.15,
        funding_rates=funding,
        rwa_yields=rwa_yields,
    )

    assert abs(decomp.factor_delta_return - 0.096) < 1e-5
    assert abs(decomp.rwa_yield_return - 0.0045) < 1e-5
