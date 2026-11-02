"""Tests for statistical volatility forecasting primitives."""

import numpy as np
import pytest

from src.models.volatility_forecasting import (
    bipower_variation,
    combine_variance_forecasts,
    ewma_variance_forecast,
    fit_volatility_model,
    forecast_price_interval,
    har_rv_forecast,
    jump_variation,
    next_period_variance,
    price_interval_from_variance,
    realized_variance,
    score_variance_forecasts,
)


def clustered_returns(size: int = 520, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.zeros(size, dtype=np.float64)
    variances = np.full(size, 0.0004, dtype=np.float64)
    for index in range(1, size):
        variances[index] = (
            0.00002 + 0.08 * returns[index - 1] ** 2 + 0.90 * variances[index - 1]
        )
        returns[index] = (
            np.sqrt(variances[index]) * rng.standard_t(7) * np.sqrt(5.0 / 7.0)
        )
    return returns


@pytest.mark.parametrize("model", ["GARCH", "EGARCH", "GJR-GARCH"])
def test_arch_family_fits_and_generates_ordered_price_ranges(model: str) -> None:
    fitted = fit_volatility_model(clustered_returns(), model=model)  # type: ignore[arg-type]
    variance = next_period_variance(fitted)
    interval = forecast_price_interval(fitted, spot_price=2_285.0, confidence=0.95)

    assert fitted.converged is True
    assert fitted.observations == 520
    assert variance > 0.0
    assert interval.lower_price < interval.median_price < interval.upper_price
    assert interval.conditional_volatility > 0.0
    assert interval.annualized_volatility > interval.conditional_volatility


def test_gjr_garch_persistence_respects_stationarity_constraint() -> None:
    fitted = fit_volatility_model(clustered_returns(), model="GJR-GARCH")
    p = fitted.parameters
    assert p["alpha"] + 0.5 * p["gamma"] + p["beta"] < 1.0


def test_student_t_range_is_wider_than_normal_range() -> None:
    student = price_interval_from_variance(
        spot_price=2_000.0, conditional_variance=0.02**2, confidence=0.95,
        distribution="student_t", degrees_of_freedom=5.0,
    )
    gaussian = price_interval_from_variance(
        spot_price=2_000.0, conditional_variance=0.02**2, confidence=0.95,
        distribution="normal", degrees_of_freedom=None,
    )
    assert student.lower_price < gaussian.lower_price
    assert student.upper_price > gaussian.upper_price


def test_ewma_and_realized_variance_estimators() -> None:
    returns = [0.01, -0.02, 0.005, -0.003, 0.009]
    assert realized_variance(returns) == pytest.approx(0.000615)
    assert bipower_variation(returns) > 0.0
    assert jump_variation(returns) >= 0.0
    assert ewma_variance_forecast(returns, decay=0.94, initial_window=3) > 0.0


def test_har_rv_forecast_uses_daily_weekly_and_monthly_features() -> None:
    rng = np.random.default_rng(2)
    daily_rv = np.exp(rng.normal(-8.0, 0.25, size=120))
    forecast = har_rv_forecast(daily_rv)

    assert forecast.forecast_variance > 0.0
    assert forecast.forecast_volatility > 0.0
    assert len(forecast.coefficients) == 4
    assert forecast.observations == 98
    assert forecast.residual_rmse >= 0.0


def test_variance_scores_and_explicit_ensemble_weights() -> None:
    score = score_variance_forecasts(
        predicted_variances=[0.9, 1.8, 3.1],
        realized_variances=[1.0, 2.0, 3.0],
    )
    assert score.observations == 3
    assert score.qlike >= 0.0
    assert score.rmse > 0.0
    assert score.mae > 0.0

    combined = combine_variance_forecasts(
        {"garch": 0.001, "har": 0.002},
        {"garch": 0.75, "har": 0.25},
    )
    assert combined == pytest.approx(0.00125)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_volatility_model([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        price_interval_from_variance(spot_price=0.0, conditional_variance=0.01)
    with pytest.raises(ValueError):
        combine_variance_forecasts({"garch": 0.01}, {"garch": 0.9})
