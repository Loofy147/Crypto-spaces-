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
from src.models.horizon_forecasting import forecast_price_interval_horizon


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
    horizon = forecast_price_interval_horizon(
        fitted, spot_price=2_285.0, horizon_steps=24,
        confidence=0.95, simulations=4_000, random_seed=17,
    )

    assert fitted.converged is True
    assert fitted.observations == 520
    assert variance > 0.0
    assert interval.lower_price < interval.median_price < interval.upper_price
    assert interval.conditional_volatility > 0.0
    assert interval.annualized_volatility > interval.conditional_volatility
    assert horizon.lower_price < horizon.median_price < horizon.upper_price
    assert horizon.horizon_steps == 24
    if model == "EGARCH":
        assert horizon.expected_cumulative_variance is None
    else:
        assert np.isfinite(horizon.expected_cumulative_variance)
        assert horizon.expected_cumulative_variance > 0.0
        if model in ("GARCH", "GJR-GARCH"):
            expected_h = next_period_variance(fitted) * 10_000.0
            expected_total = 0.0
            for _ in range(24):
                expected_total += expected_h
                if model == "GARCH":
                    persistence = fitted.parameters["alpha"] + fitted.parameters["beta"]
                else:
                    persistence = (
                        fitted.parameters["alpha"]
                        + 0.5 * fitted.parameters["gamma"]
                        + fitted.parameters["beta"]
                    )
                expected_h = fitted.parameters["omega"] + persistence * expected_h
            assert horizon.expected_cumulative_variance == pytest.approx(
                expected_total / 10_000.0, rel=1e-12
            )
    assert horizon.simulations == 4_000
    assert horizon.conditional_volatility > 0.0
    assert horizon.conditional_volatility < 1.0
    assert horizon.annualized_volatility > horizon.conditional_volatility


def test_egarch_normal_horizon_has_finite_expected_variance() -> None:
    fitted = fit_volatility_model(
        clustered_returns(), model="EGARCH", distribution="normal"
    )
    horizon = forecast_price_interval_horizon(
        fitted, 2_000.0, simulations=2_000, random_seed=101
    )
    assert horizon.expected_cumulative_variance is not None
    assert np.isfinite(horizon.expected_cumulative_variance)
    assert horizon.expected_cumulative_variance > 0.0


def test_egarch_normal_expected_variance_matches_closed_form_two_step() -> None:
    from scipy.special import ndtr

    fitted = fit_volatility_model(
        clustered_returns(), model="EGARCH", distribution="normal"
    )
    p = fitted.parameters
    initial = next_period_variance(fitted) * 10_000.0
    expected_abs = np.sqrt(2.0 / np.pi)
    plus = p["alpha"] + p["gamma"]
    minus = p["alpha"] - p["gamma"]
    innovation_moment = np.exp(-p["alpha"] * expected_abs) * (
        np.exp(0.5 * plus**2) * ndtr(plus)
        + np.exp(0.5 * minus**2) * ndtr(minus)
    )
    second = np.exp(p["omega"]) * initial**p["beta"] * innovation_moment
    expected_two_step = (initial + second) / 10_000.0

    horizon = forecast_price_interval_horizon(
        fitted, 2_000.0, horizon_steps=2, simulations=2_000, random_seed=103
    )
    assert horizon.expected_cumulative_variance == pytest.approx(
        expected_two_step, rel=1e-12
    )


def test_horizon_simulation_is_reproducible() -> None:
    fitted = fit_volatility_model(clustered_returns(), model="GARCH")
    first = forecast_price_interval_horizon(
        fitted, 2_000.0, simulations=2_000, random_seed=99
    )
    second = forecast_price_interval_horizon(
        fitted, 2_000.0, simulations=2_000, random_seed=99
    )
    assert first.lower_price == pytest.approx(second.lower_price)
    assert first.median_price == pytest.approx(second.median_price)
    assert first.upper_price == pytest.approx(second.upper_price)


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
    fitted = fit_volatility_model(clustered_returns(), model="GARCH")
    with pytest.raises(ValueError, match="horizon_steps"):
        forecast_price_interval_horizon(fitted, 2_000.0, horizon_steps=0)
