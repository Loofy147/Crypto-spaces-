"""Immutable State Engine & Pure Reducers for Event Sourcing."""

from typing import Iterable, List
from src.core.events import (
    DerivativesMetricsEvent,
    MarketTickEvent,
    OrderBookDepthEvent,
    OrderExecutedEvent,
    PortfolioEvent,
    RebalanceTriggeredEvent,
    YieldAccruedEvent,
)
from src.core.types import Position, PortfolioState


def apply_market_tick(state: PortfolioState, event: MarketTickEvent) -> PortfolioState:
    """Pure reducer for MarketTickEvent."""
    positions = dict(state.positions)
    if event.symbol in positions:
        old_pos = positions[event.symbol]
        new_notional = old_pos.units * event.price
        positions[event.symbol] = Position(
            asset_symbol=event.symbol,
            units=old_pos.units,
            current_price=event.price,
            notional_value=new_notional,
            weight=0.0,  # recalculated below
            accrued_yield=old_pos.accrued_yield,
        )

    # Recalculate total equity and weights
    total_position_val = sum(p.notional_value for p in positions.values())
    total_equity = state.cash_balance + total_position_val

    updated_positions = {}
    for sym, pos in positions.items():
        w = pos.notional_value / total_equity if total_equity > 0 else 0.0
        updated_positions[sym] = Position(
            asset_symbol=pos.asset_symbol,
            units=pos.units,
            current_price=pos.current_price,
            notional_value=pos.notional_value,
            weight=w,
            accrued_yield=pos.accrued_yield,
        )

    return PortfolioState(
        timestamp=max(state.timestamp, event.timestamp),
        positions=updated_positions,
        cash_balance=state.cash_balance,
        total_equity=total_equity,
        unrealized_pnl=state.unrealized_pnl,
        margin_used=state.margin_used,
        event_sequence=event.sequence if event.sequence > 0 else state.event_sequence + 1,
    )


def apply_order_executed(state: PortfolioState, event: OrderExecutedEvent) -> PortfolioState:
    """Pure reducer for OrderExecutedEvent."""
    positions = dict(state.positions)
    current_pos = positions.get(
        event.symbol,
        Position(asset_symbol=event.symbol, units=0.0, current_price=event.execution_price, notional_value=0.0),
    )

    trade_notional = event.units * event.execution_price
    if event.side == "BUY":
        new_units = current_pos.units + event.units
        new_cash = state.cash_balance - trade_notional - event.fee
    else:  # SELL
        new_units = current_pos.units - event.units
        new_cash = state.cash_balance + trade_notional - event.fee

    new_notional = new_units * event.execution_price
    positions[event.symbol] = Position(
        asset_symbol=event.symbol,
        units=new_units,
        current_price=event.execution_price,
        notional_value=new_notional,
        weight=0.0,
        accrued_yield=current_pos.accrued_yield,
    )

    total_position_val = sum(p.notional_value for p in positions.values())
    total_equity = new_cash + total_position_val

    updated_positions = {}
    for sym, pos in positions.items():
        w = pos.notional_value / total_equity if total_equity > 0 else 0.0
        updated_positions[sym] = Position(
            asset_symbol=pos.asset_symbol,
            units=pos.units,
            current_price=pos.current_price,
            notional_value=pos.notional_value,
            weight=w,
            accrued_yield=pos.accrued_yield,
        )

    return PortfolioState(
        timestamp=max(state.timestamp, event.timestamp),
        positions=updated_positions,
        cash_balance=new_cash,
        total_equity=total_equity,
        unrealized_pnl=state.unrealized_pnl,
        margin_used=state.margin_used,
        event_sequence=event.sequence if event.sequence > 0 else state.event_sequence + 1,
    )


def apply_yield_accrued(state: PortfolioState, event: YieldAccruedEvent) -> PortfolioState:
    """Pure reducer for YieldAccruedEvent."""
    positions = dict(state.positions)
    if event.asset_symbol in positions:
        pos = positions[event.asset_symbol]
        new_accrued = pos.accrued_yield + event.amount
        positions[event.asset_symbol] = Position(
            asset_symbol=pos.asset_symbol,
            units=pos.units,
            current_price=pos.current_price,
            notional_value=pos.notional_value,
            weight=pos.weight,
            accrued_yield=new_accrued,
        )

    new_cash = state.cash_balance + event.amount
    total_position_val = sum(p.notional_value for p in positions.values())
    total_equity = new_cash + total_position_val

    return PortfolioState(
        timestamp=max(state.timestamp, event.timestamp),
        positions=positions,
        cash_balance=new_cash,
        total_equity=total_equity,
        unrealized_pnl=state.unrealized_pnl,
        margin_used=state.margin_used,
        event_sequence=event.sequence if event.sequence > 0 else state.event_sequence + 1,
    )


def apply_event(state: PortfolioState, event: PortfolioEvent) -> PortfolioState:
    """Pure dispatcher reducer for all events."""
    if isinstance(event, MarketTickEvent):
        return apply_market_tick(state, event)
    elif isinstance(event, OrderExecutedEvent):
        return apply_order_executed(state, event)
    elif isinstance(event, YieldAccruedEvent):
        return apply_yield_accrued(state, event)
    elif isinstance(event, (OrderBookDepthEvent, DerivativesMetricsEvent, RebalanceTriggeredEvent)):
        # Pass through state with updated timestamp & sequence
        return PortfolioState(
            timestamp=max(state.timestamp, event.timestamp),
            positions=state.positions,
            cash_balance=state.cash_balance,
            total_equity=state.total_equity,
            unrealized_pnl=state.unrealized_pnl,
            margin_used=state.margin_used,
            event_sequence=event.sequence if event.sequence > 0 else state.event_sequence + 1,
        )
    return state


def reduce_events(initial_state: PortfolioState, events: Iterable[PortfolioEvent]) -> PortfolioState:
    """Point-in-time state reconstruction from an append-only event log."""
    state = initial_state
    for event in events:
        state = apply_event(state, event)
    return state
