"""Unit tests for src/core module."""

import pytest
from src.core.events import (
    MarketTickEvent,
    OrderExecutedEvent,
    YieldAccruedEvent,
)
from src.core.state import apply_event, reduce_events
from src.core.types import AssetCategory, Position, PortfolioState


def test_portfolio_state_immutability() -> None:
    state = PortfolioState(timestamp=100.0, cash_balance=1000.0, total_equity=1000.0)
    with pytest.raises(Exception):
        state.cash_balance = 2000.0  # type: ignore[misc]


def test_market_tick_event_reducer() -> None:
    initial_state = PortfolioState(
        timestamp=0.0,
        positions={
            "BTC": Position(
                asset_symbol="BTC", units=1.0, current_price=50000.0, notional_value=50000.0, weight=0.5
            )
        },
        cash_balance=50000.0,
        total_equity=100000.0,
    )

    tick = MarketTickEvent(
        event_id="t1",
        timestamp=10.0,
        symbol="BTC",
        price=60000.0,
    )

    new_state = apply_event(initial_state, tick)

    assert new_state.timestamp == 10.0
    assert new_state.positions["BTC"].current_price == 60000.0
    assert new_state.positions["BTC"].notional_value == 60000.0
    assert new_state.cash_balance == 50000.0
    assert new_state.total_equity == 110000.0
    assert new_state.positions["BTC"].weight == 60000.0 / 110000.0


def test_order_executed_event_reducer() -> None:
    initial_state = PortfolioState(
        timestamp=0.0,
        positions={},
        cash_balance=100000.0,
        total_equity=100000.0,
    )

    buy_order = OrderExecutedEvent(
        event_id="o1",
        timestamp=5.0,
        order_id="ord_1",
        symbol="ETH",
        side="BUY",
        units=10.0,
        execution_price=3000.0,
        fee=10.0,
    )

    state_after_buy = apply_event(initial_state, buy_order)

    assert state_after_buy.positions["ETH"].units == 10.0
    assert state_after_buy.positions["ETH"].notional_value == 30000.0
    assert state_after_buy.cash_balance == 100000.0 - 30000.0 - 10.0
    assert state_after_buy.total_equity == 100000.0 - 10.0


def test_yield_accrued_event_reducer() -> None:
    initial_state = PortfolioState(
        timestamp=0.0,
        positions={
            "BUIDL": Position(
                asset_symbol="BUIDL", units=100000.0, current_price=1.0, notional_value=100000.0
            )
        },
        cash_balance=0.0,
        total_equity=100000.0,
    )

    yield_event = YieldAccruedEvent(
        event_id="y1",
        timestamp=86400.0,
        asset_symbol="BUIDL",
        amount=12.33,
        apy_rate=0.045,
    )

    new_state = apply_event(initial_state, yield_event)

    assert new_state.positions["BUIDL"].accrued_yield == 12.33
    assert new_state.cash_balance == 12.33
    assert new_state.total_equity == 100012.33


def test_reduce_events_sequence() -> None:
    initial_state = PortfolioState(timestamp=0.0, cash_balance=100000.0, total_equity=100000.0)

    events = [
        OrderExecutedEvent(
            event_id="o1",
            timestamp=1.0,
            order_id="1",
            symbol="BTC",
            side="BUY",
            units=1.0,
            execution_price=40000.0,
            fee=0.0,
        ),
        MarketTickEvent(event_id="t1", timestamp=2.0, symbol="BTC", price=42000.0),
    ]

    final_state = reduce_events(initial_state, events)

    assert final_state.timestamp == 2.0
    assert final_state.positions["BTC"].notional_value == 42000.0
    assert final_state.cash_balance == 60000.0
    assert final_state.total_equity == 102000.0
