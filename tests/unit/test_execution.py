"""Unit tests for src/execution module."""

from src.core.types import AssetCategory, Position, PortfolioState
from src.execution.pre_trade_risk import PreTradeRiskEngine
from src.execution.rebalancer import HybridDriftRebalancer
from src.execution.smart_router import SmartOrderRouter


def test_smart_router_cost_estimation() -> None:
    router = SmartOrderRouter(base_fee_bps=5.0, half_spread_bps=2.5)
    estimate = router.estimate_trade_cost(
        symbol="BTC",
        trade_notional=100000.0,
        depth_2pct_notional=1000000.0,
        tracking_error_reduction=500.0,
    )

    assert estimate.spread_cost == 25.0  # 2.5 bps of 100k
    assert estimate.fee_cost == 50.0  # 5 bps of 100k
    assert estimate.total_execution_cost > 75.0
    assert estimate.is_cost_effective is True


def test_hybrid_drift_rebalancer() -> None:
    rebalancer = HybridDriftRebalancer(core_drift_threshold=0.05, satellite_drift_threshold=0.10)

    current_w = {"BTC": 0.50, "ETH": 0.20, "SOL": 0.30}  # BTC core drifted from 0.40 to 0.50 (+10% > 5%)
    target_w = {"BTC": 0.40, "ETH": 0.20, "SOL": 0.40}
    categories = {"BTC": AssetCategory.CORE, "ETH": AssetCategory.CORE, "SOL": AssetCategory.SATELLITE}
    depth_map = {"BTC": 5000000.0, "ETH": 2000000.0, "SOL": 1000000.0}

    decision = rebalancer.evaluate_rebalance(
        current_weights=current_w,
        target_weights=target_w,
        asset_categories=categories,
        total_equity=1_000_000.0,
        depth_2pct_map=depth_map,
    )

    assert decision.should_rebalance is True
    assert "BTC" in decision.triggered_assets


def test_pre_trade_risk_engine() -> None:
    risk_engine = PreTradeRiskEngine(max_satellite_single_cap=0.15)
    state = PortfolioState(
        timestamp=0.0,
        cash_balance=100000.0,
        positions={
            "BTC": Position(
                asset_symbol="BTC", units=1.0, current_price=50000.0, notional_value=50000.0
            ),
            "BUIDL": Position(
                asset_symbol="BUIDL", units=50000.0, current_price=1.0, notional_value=50000.0
            ),
        },
    )
    categories = {"BTC": AssetCategory.CORE, "BUIDL": AssetCategory.RWA_YIELD, "SOL": AssetCategory.SATELLITE}

    # Effective collateral: 100k cash + 50k*0.85 (BTC 15% haircut) + 50k*0.99 (BUIDL 1% haircut)
    effective_collateral = risk_engine.compute_effective_collateral(state, categories)
    assert effective_collateral == 100000.0 + 42500.0 + 49500.0

    # Test concentration cap violation
    proposed_w = {"BTC": 0.40, "ETH": 0.20, "SOL": 0.25, "BUIDL": 0.15}  # SOL 25% > 15% cap
    res = risk_engine.validate_pre_trade(state, proposed_w, categories)

    assert res.passed is False
    assert any("Concentration cap breached" in v for v in res.violations)
