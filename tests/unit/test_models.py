"""Unit tests for src/models module."""

import numpy as np
from src.models.regime_classifier import MarketRegime, RegimeClassifier
from src.models.risk_parity import RiskParityOptimizer, compute_marginal_risk_contributions
from src.models.rwa_sleeve import RWASleeveManager
from src.models.vol_targeting import VolatilityTargeter


def test_rwa_sleeve_initial_allocation() -> None:
    manager = RWASleeveManager()
    satellites = ["SOL", "AVAX"]
    depth_map = {"SOL": 20000.0, "AVAX": 5000.0}  # AVAX below 10000 threshold

    plan = manager.compute_initial_target_weights(
        satellite_assets=satellites,
        depth_2pct_map=depth_map,
        min_depth_threshold=10000.0,
    )

    assert plan.weights["BTC"] == 0.40
    assert plan.weights["ETH"] == 0.20
    assert "SOL" in plan.weights
    assert "AVAX" not in plan.weights
    assert plan.weights["BUIDL"] == 0.10 + 0.15  # 10% base + 15% unallocated satellite from AVAX

    daily_y = manager.calculate_daily_yield(100000.0)
    assert abs(daily_y - (100000.0 * 0.045 / 365.0)) < 1e-5


def test_risk_parity_optimizer() -> None:
    optimizer = RiskParityOptimizer()
    assets = ["BTC", "ETH", "SOL"]

    cov = np.array(
        [
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16],
        ]
    )

    weights = optimizer.optimize_risk_parity(assets, cov)

    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert weights["SOL"] < weights["BTC"]

    # Test edge cases (1 asset, zero vol)
    single_w = optimizer.optimize_risk_parity(["BTC"], np.array([[0.04]]))
    assert single_w == {"BTC": 1.0}

    mcr, rc = compute_marginal_risk_contributions(np.array([0.5, 0.5]), np.zeros((2, 2)))
    assert np.all(mcr == 0.0)


def test_volatility_targeter() -> None:
    targeter = VolatilityTargeter(target_volatility=0.20)
    base_weights = {"BTC": 0.40, "ETH": 0.20, "SOL": 0.30, "BUIDL": 0.10}
    realized_vols = {"BTC": 0.30, "ETH": 0.40, "SOL": 0.60, "BUIDL": 0.0}

    res = targeter.scale_positions(base_weights, realized_vols)

    assert res.scaler_factor < 1.0
    assert res.scaled_weights["SOL"] < base_weights["SOL"]
    assert res.rwa_sleeve_weight > base_weights["BUIDL"]


def test_regime_classifier() -> None:
    classifier = RegimeClassifier()
    bull_res = classifier.classify([100.0, 1.5, 0.70, 0.20])
    assert bull_res.regime == MarketRegime.BULL
    assert bull_res.risk_aversion_gamma == 1.0

    bear_res = classifier.classify([-100.0, 0.5, 0.10, -0.20])
    assert bear_res.regime == MarketRegime.BEAR
    assert bear_res.risk_aversion_gamma == 3.0

    # Train model
    X = np.array(
        [
            [100.0, 1.5, 0.70, 0.20],
            [-100.0, 0.5, 0.10, -0.20],
            [0.0, 1.0, 0.40, 0.0],
        ]
    )
    y = np.array([0, 1, 2])
    classifier.train(X, y)

    res_trained = classifier.classify([100.0, 1.5, 0.70, 0.20])
    assert res_trained.regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.CONSOLIDATION]

    # Test train with empty data
    empty_classifier = RegimeClassifier()
    empty_classifier.train(np.array([]), np.array([]))
    assert empty_classifier._is_trained is False

    # Test fallback classification regimes
    cons_res = classifier.classify([0.0, 1.0, 0.40, 0.0])
    assert cons_res.risk_aversion_gamma in [1.0, 2.0, 3.0]


def test_risk_parity_edge_cases_and_scipy_fallback() -> None:
    optimizer = RiskParityOptimizer()

    # Empty asset names
    assert optimizer.optimize_risk_parity([], np.array([[]])) == {}

    # SciPy SLSQP Direct Fallback
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    scipy_weights = optimizer._solve_scipy(2, cov)
    assert scipy_weights is not None
    assert abs(np.sum(scipy_weights) - 1.0) < 1e-4

    # Covariance matrix causing CVXPY failure fallback
    bad_cov = np.array([[1e-12, 0.0], [0.0, 1e-12]])
    res = optimizer.optimize_risk_parity(["A", "B"], bad_cov)
    assert "A" in res and "B" in res


def test_volatility_targeter_with_cov_matrix() -> None:
    targeter = VolatilityTargeter(target_volatility=0.20)
    cov = np.array([[0.09, 0.02], [0.02, 0.16]])
    assets = ["BTC", "ETH"]
    base_w = {"BTC": 0.50, "ETH": 0.50}

    res = targeter.scale_positions(
        base_weights=base_w,
        realized_vols={},
        portfolio_cov_matrix=cov,
        asset_names=assets,
    )

    assert res.scaler_factor < 1.0
    assert res.realized_volatility > 0.0

    # Zero portfolio vol
    res_zero = targeter.scale_positions(
        base_weights={"BTC": 0.0},
        realized_vols={"BTC": 0.0},
    )
    assert res_zero.scaler_factor == 1.0
