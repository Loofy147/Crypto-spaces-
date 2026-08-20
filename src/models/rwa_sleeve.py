"""60/30/10 Core-Satellite Architecture & RWA Yield Sleeve Manager."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CoreSatelliteTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    core_target: float = 0.60  # BTC 40%, ETH 20%
    satellite_target: float = 0.30  # Filtered by min 2% depth
    rwa_yield_target: float = 0.10  # Tokenized Treasuries (BUIDL / USDY)

    btc_target: float = 0.40
    eth_target: float = 0.20


class RWASleeveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_symbol: str = "BUIDL"  # BUIDL or USDY
    base_apy: float = 0.045  # 4.5% annual yield
    haircut: float = 0.01  # 1% collateral haircut


class AllocationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: Dict[str, float]
    core_weight: float
    satellite_weight: float
    rwa_weight: float


class RWASleeveManager:
    """Manages 60/30/10 Core-Satellite targets and RWA Yield Overlay."""

    def __init__(
        self,
        targets: CoreSatelliteTarget = CoreSatelliteTarget(),
        rwa_config: RWASleeveConfig = RWASleeveConfig(),
    ) -> None:
        self.targets = targets
        self.rwa_config = rwa_config

    def compute_initial_target_weights(
        self,
        satellite_assets: List[str],
        depth_2pct_map: Dict[str, float],
        min_depth_threshold: float = 10000.0,
    ) -> AllocationPlan:
        """Filter satellite assets by minimum 2% order book depth and allocate weights."""
        eligible_satellites = [
            asset
            for asset in satellite_assets
            if depth_2pct_map.get(asset, 0.0) >= min_depth_threshold
        ]

        weights: Dict[str, float] = {}
        weights["BTC"] = self.targets.btc_target
        weights["ETH"] = self.targets.eth_target

        num_candidates = len(satellite_assets)
        if num_candidates > 0:
            target_per_candidate = self.targets.satellite_target / num_candidates
            for asset in eligible_satellites:
                weights[asset] = target_per_candidate
            sat_total = target_per_candidate * len(eligible_satellites)
        else:
            sat_total = 0.0

        # Any unallocated satellite weight goes into RWA yield sleeve
        rwa_total = self.targets.rwa_yield_target + (self.targets.satellite_target - sat_total)
        weights[self.rwa_config.asset_symbol] = rwa_total

        return AllocationPlan(
            weights=weights,
            core_weight=self.targets.core_target,
            satellite_weight=sat_total,
            rwa_weight=rwa_total,
        )

    def calculate_daily_yield(self, rwa_notional: float, custom_apy: Optional[float] = None) -> float:
        """Calculate daily accrued yield from tokenized Treasury holdings."""
        apy = custom_apy if custom_apy is not None else self.rwa_config.base_apy
        daily_rate = apy / 365.0
        return rwa_notional * daily_rate
