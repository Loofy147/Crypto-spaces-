"""Backtest Validation & Overfitting Verification Gate Tests."""

import numpy as np
import pytest
from src.backtest.dsr import (
    DeflatedSharpeRatioCalculator,
    DSRResult,
    StrategyOverfittedException,
)


def test_overfitting_gate_passes_for_valid_strategy() -> None:
    valid_result = DSRResult(
        dsr_score=0.98,
        pbo_score=0.05,
        estimated_sharpe=2.1,
        benchmark_sharpe_star=1.2,
        skewness=0.1,
        kurtosis=3.0,
        num_trials=10,
        num_observations=500,
        is_valid=True,
    )

    # Should not raise exception
    DeflatedSharpeRatioCalculator.validate_overfitting_gate(valid_result)


def test_overfitting_gate_fails_pbo_threshold() -> None:
    invalid_pbo_result = DSRResult(
        dsr_score=0.98,
        pbo_score=0.15,  # Breaches 0.10 threshold!
        estimated_sharpe=2.1,
        benchmark_sharpe_star=1.2,
        skewness=0.1,
        kurtosis=3.0,
        num_trials=10,
        num_observations=500,
        is_valid=False,
    )

    with pytest.raises(StrategyOverfittedException) as exc_info:
        DeflatedSharpeRatioCalculator.validate_overfitting_gate(invalid_pbo_result)

    assert "PBO=0.1500 (must be < 0.10)" in str(exc_info.value)


def test_overfitting_gate_fails_dsr_threshold() -> None:
    invalid_dsr_result = DSRResult(
        dsr_score=0.85,  # Breaches 0.95 threshold!
        pbo_score=0.02,
        estimated_sharpe=1.1,
        benchmark_sharpe_star=1.2,
        skewness=-0.5,
        kurtosis=4.5,
        num_trials=50,
        num_observations=200,
        is_valid=False,
    )

    with pytest.raises(StrategyOverfittedException) as exc_info:
        DeflatedSharpeRatioCalculator.validate_overfitting_gate(invalid_dsr_result)

    assert "DSR=0.8500 (must be >= 0.95)" in str(exc_info.value)
