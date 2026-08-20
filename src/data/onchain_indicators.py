"""On-chain indicators calculator (NUPL, Realized Cap, Hot Capital, ASI)."""

from enum import Enum
from typing import Dict, List, Sequence
import polars as pl
from pydantic import BaseModel, ConfigDict


class ASIRegime(str, Enum):
    BITCOIN_DOMINANCE = "BITCOIN_DOMINANCE"
    TRANSITIONAL_ROTATION = "TRANSITIONAL_ROTATION"
    ALTSEASON = "ALTSEASON"


class ASIMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    asi_score: float  # 0 to 100
    outperforming_count: int
    total_universe_count: int
    regime: ASIRegime


class OnChainMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    nupl: float
    realized_cap_change_pct: float
    hot_capital_share: float


class OnChainIndicatorsCalculator:
    """Calculates on-chain valuation and capital rotation indicators."""

    @staticmethod
    def calculate_onchain_metrics(
        market_cap: float,
        realized_cap: float,
        prev_realized_cap_30d: float,
        hot_capital_val: float,
    ) -> OnChainMetrics:
        """Computes NUPL, 30-day Realized Cap Change %, and Hot Capital Share."""
        if market_cap <= 0:
            nupl = 0.0
        else:
            nupl = (market_cap - realized_cap) / market_cap

        if prev_realized_cap_30d <= 0:
            realized_cap_change = 0.0
        else:
            realized_cap_change = ((realized_cap - prev_realized_cap_30d) / prev_realized_cap_30d) * 100.0

        if market_cap <= 0:
            hot_cap_share = 0.0
        else:
            hot_cap_share = hot_capital_val / market_cap

        return OnChainMetrics(
            nupl=nupl,
            realized_cap_change_pct=realized_cap_change,
            hot_capital_share=hot_cap_share,
        )

    @staticmethod
    def calculate_altcoin_season_index(
        btc_90d_return: float,
        altcoin_90d_returns: Dict[str, float],
    ) -> ASIMetrics:
        """Calculates Altcoin Season Index (ASI) over rolling 90-day window.

        Excludes stablecoins and wrapped assets by caller filter.
        """
        if not altcoin_90d_returns:
            return ASIMetrics(
                asi_score=0.0,
                outperforming_count=0,
                total_universe_count=0,
                regime=ASIRegime.BITCOIN_DOMINANCE,
            )

        outperforming = sum(
            1 for ret in altcoin_90d_returns.values() if ret > btc_90d_return
        )
        total = len(altcoin_90d_returns)
        score = (outperforming / total) * 100.0

        if score < 25.0:
            regime = ASIRegime.BITCOIN_DOMINANCE
        elif score <= 75.0:
            regime = ASIRegime.TRANSITIONAL_ROTATION
        else:
            regime = ASIRegime.ALTSEASON

        return ASIMetrics(
            asi_score=score,
            outperforming_count=outperforming,
            total_universe_count=total,
            regime=regime,
        )
