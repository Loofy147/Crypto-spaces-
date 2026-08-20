"""Domain types for institutional digital asset portfolio management."""

from enum import Enum
from typing import Dict, NamedTuple
from pydantic import BaseModel, ConfigDict, Field


class Currency(str, Enum):
    USD = "USD"
    USDC = "USDC"
    USDT = "USDT"
    BTC = "BTC"
    ETH = "ETH"


class AssetCategory(str, Enum):
    CORE = "CORE"
    SATELLITE = "SATELLITE"
    RWA_YIELD = "RWA_YIELD"
    CASH = "CASH"


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    category: AssetCategory
    haircut: float = Field(ge=0.0, le=1.0)
    is_derivative: bool = False
    underlying_symbol: str | None = None


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_symbol: str
    units: float
    current_price: float
    notional_value: float
    weight: float = 0.0
    accrued_yield: float = 0.0


class PortfolioState(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: float
    positions: Dict[str, Position] = Field(default_factory=dict)
    cash_balance: float = 0.0
    total_equity: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    event_sequence: int = 0
