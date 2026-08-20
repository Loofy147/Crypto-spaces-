"""Hybrid Drift Control & Rebalancing Engine."""

from typing import Dict, List
from pydantic import BaseModel, ConfigDict
from src.core.types import AssetCategory
from src.execution.smart_router import SmartOrderRouter, TradeCostEstimate


class RebalanceInstruction(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    current_weight: float
    target_weight: float
    drift: float
    target_notional_change: float
    cost_estimate: TradeCostEstimate
    approved: bool
    rejection_reason: str | None = None


class RebalanceDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    should_rebalance: bool
    triggered_assets: List[str]
    instructions: List[RebalanceInstruction]
    total_estimated_cost: float


class HybridDriftRebalancer:
    """Monitors drift bands (±5% core, ±10% satellite) and filters execution by cost-benefit ratio."""

    def __init__(
        self,
        core_drift_threshold: float = 0.05,
        satellite_drift_threshold: float = 0.10,
        smart_router: SmartOrderRouter | None = None,
    ) -> None:
        self.core_drift_threshold = core_drift_threshold
        self.satellite_drift_threshold = satellite_drift_threshold
        self.router = smart_router or SmartOrderRouter()

    def evaluate_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        asset_categories: Dict[str, AssetCategory],
        total_equity: float,
        depth_2pct_map: Dict[str, float],
    ) -> RebalanceDecision:
        """Evaluates absolute weight drift and generates cost-filtered trade instructions."""
        all_symbols = set(current_weights.keys()).union(set(target_weights.keys()))
        triggered_assets: List[str] = []
        instructions: List[RebalanceInstruction] = []
        total_cost = 0.0

        for symbol in all_symbols:
            curr_w = current_weights.get(symbol, 0.0)
            targ_w = target_weights.get(symbol, 0.0)
            drift = abs(curr_w - targ_w)

            category = asset_categories.get(symbol, AssetCategory.SATELLITE)
            threshold = (
                self.core_drift_threshold
                if category == AssetCategory.CORE
                else self.satellite_drift_threshold
            )

            is_breached = drift > threshold
            if is_breached:
                triggered_assets.append(symbol)

            notional_change = (targ_w - curr_w) * total_equity
            depth_2pct = depth_2pct_map.get(symbol, 100000.0)

            # Estimated tracking error reduction (quadratic loss of weight drift)
            tracking_error_reduction = 0.5 * (drift**2) * total_equity

            cost_est = self.router.estimate_trade_cost(
                symbol=symbol,
                trade_notional=notional_change,
                depth_2pct_notional=depth_2pct,
                tracking_error_reduction=tracking_error_reduction,
            )

            approved = is_breached and cost_est.is_cost_effective
            reason = None
            if not is_breached:
                reason = "Drift within tolerance band"
            elif not cost_est.is_cost_effective:
                reason = f"Execution cost ({cost_est.total_execution_cost:.2f}) exceeds tracking error reduction ({tracking_error_reduction:.2f})"

            if approved:
                total_cost += cost_est.total_execution_cost

            instructions.append(
                RebalanceInstruction(
                    symbol=symbol,
                    current_weight=curr_w,
                    target_weight=targ_w,
                    drift=drift,
                    target_notional_change=notional_change,
                    cost_estimate=cost_est,
                    approved=approved,
                    rejection_reason=reason,
                )
            )

        should_rebal = len(triggered_assets) > 0 and any(i.approved for i in instructions)

        return RebalanceDecision(
            should_rebalance=should_rebal,
            triggered_assets=triggered_assets,
            instructions=instructions,
            total_estimated_cost=total_cost,
        )
