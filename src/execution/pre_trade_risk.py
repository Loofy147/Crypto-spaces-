"""Pre-Trade Risk Engine, Collateral Haircuts, and Liquidation Risk Checks."""

from typing import Dict, List
from pydantic import BaseModel, ConfigDict
from src.core.types import AssetCategory, PortfolioState


class RiskCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    effective_collateral_value: float
    required_margin: float
    margin_coverage_ratio: float
    violations: List[str]


class PreTradeRiskEngine:
    """Enforces multi-collateral haircuts, concentration caps, and liquidation buffers."""

    DEFAULT_HAIR_CUTS: Dict[AssetCategory, float] = {
        AssetCategory.CASH: 0.0,
        AssetCategory.RWA_YIELD: 0.01,  # 1% Haircut for BUIDL/USDY
        AssetCategory.CORE: 0.15,  # 15% Haircut for BTC/ETH
        AssetCategory.SATELLITE: 0.40,  # 40% Haircut for Satellite Altcoins
    }

    def __init__(
        self,
        haircuts: Dict[AssetCategory, float] | None = None,
        max_satellite_single_cap: float = 0.15,  # Max 15% in any single satellite
        min_margin_coverage: float = 1.25,  # 125% collateral buffer
    ) -> None:
        self.haircuts = haircuts or self.DEFAULT_HAIR_CUTS
        self.max_satellite_single_cap = max_satellite_single_cap
        self.min_margin_coverage = min_margin_coverage

    def compute_effective_collateral(
        self,
        state: PortfolioState,
        asset_categories: Dict[str, AssetCategory],
    ) -> float:
        """Calculates effective collateral value post-haircut."""
        total_collateral = state.cash_balance  # Cash has 0 haircut

        for symbol, pos in state.positions.items():
            cat = asset_categories.get(symbol, AssetCategory.SATELLITE)
            haircut = self.haircuts.get(cat, 0.40)
            effective_val = pos.notional_value * (1.0 - haircut)
            total_collateral += effective_val

        return total_collateral

    def validate_pre_trade(
        self,
        state: PortfolioState,
        proposed_weights: Dict[str, float],
        asset_categories: Dict[str, AssetCategory],
        required_margin: float = 0.0,
    ) -> RiskCheckResult:
        """Validates proposed portfolio allocations against collateral, margin, and caps."""
        violations: List[str] = []

        # 1. Concentration Caps
        for symbol, weight in proposed_weights.items():
            cat = asset_categories.get(symbol, AssetCategory.SATELLITE)
            if cat == AssetCategory.SATELLITE and weight > self.max_satellite_single_cap:
                violations.append(
                    f"Concentration cap breached for {symbol}: proposed weight {weight:.2%} > max satellite cap {self.max_satellite_single_cap:.2%}"
                )

        # 2. Total Weight Sum Check
        total_w = sum(proposed_weights.values())
        if abs(total_w - 1.0) > 1e-4 and total_w > 0:
            violations.append(f"Proposed weights sum to {total_w:.4f}, expected 1.0")

        # 3. Collateral & Liquidation Coverage Check
        effective_collateral = self.compute_effective_collateral(state, asset_categories)

        if required_margin > 0.0:
            coverage_ratio = effective_collateral / required_margin
            if coverage_ratio < self.min_margin_coverage:
                violations.append(
                    f"Liquidation risk: Margin coverage ratio {coverage_ratio:.2f} below required buffer {self.min_margin_coverage:.2f}"
                )
        else:
            coverage_ratio = float("inf") if effective_collateral > 0 else 0.0

        passed = len(violations) == 0

        return RiskCheckResult(
            passed=passed,
            effective_collateral_value=effective_collateral,
            required_margin=required_margin,
            margin_coverage_ratio=coverage_ratio,
            violations=violations,
        )
