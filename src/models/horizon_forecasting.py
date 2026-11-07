"""Multi-step conditional price scenarios for fitted ARCH-family models.

The horizon interval is produced by path simulation. It is a conditional
model range, not a directional forecast or a guaranteed trading range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volatility_forecasting import (
    VolatilityModelFit,
    _expected_abs_standardized_t,
    next_period_variance,
)


@dataclass(frozen=True)
class HorizonPriceInterval:
    spot_price: float
    confidence: float
    lower_price: float
    median_price: float
    upper_price: float
    horizon_steps: int
    simulations: int
    expected_cumulative_variance: float | None
    conditional_volatility: float
    annualized_volatility: float
    mean_log_return: float
    degrees_of_freedom: float | None


def forecast_price_interval_horizon(
    fit: VolatilityModelFit,
    spot_price: float,
    *,
    horizon_steps: int = 24,
    confidence: float = 0.95,
    simulations: int = 20_000,
    random_seed: int = 1_729,
    periods_per_year: int = 365 * 24,
) -> HorizonPriceInterval:
    """Simulate conditional return paths and return a horizon price interval.

    Model parameters and filtered variances are expressed in percentage-point
    units by the estimator. Simulated cumulative log returns are converted
    back to decimal units before being applied to the spot price.
    """
    if not np.isfinite(spot_price) or spot_price <= 0:
        raise ValueError("spot_price must be positive and finite")
    if not isinstance(horizon_steps, int) or not 1 <= horizon_steps <= 168:
        raise ValueError("horizon_steps must be an integer between 1 and 168")
    if not 0.5 < confidence < 0.999:
        raise ValueError("confidence must be between 0.5 and 0.999")
    if not isinstance(simulations, int) or simulations < 1_000:
        raise ValueError("simulations must be an integer of at least 1000")
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    parameters = fit.parameters
    degrees_of_freedom = parameters.get("nu")
    if fit.distribution == "student_t":
        if degrees_of_freedom is None or degrees_of_freedom <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed 2")
        innovation_scale = float(np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom))
        expected_abs = _expected_abs_standardized_t(degrees_of_freedom)
    else:
        innovation_scale = 1.0
        expected_abs = float(np.sqrt(2.0 / np.pi))

    rng = np.random.default_rng(random_seed)
    variance_pct2 = np.full(
        simulations, next_period_variance(fit) * 10_000.0, dtype=np.float64
    )
    cumulative_log_returns = np.zeros(simulations, dtype=np.float64)
    cumulative_variance_pct2 = np.zeros(simulations, dtype=np.float64)
    mu = float(parameters["mu"])

    for _ in range(horizon_steps):
        if fit.distribution == "student_t":
            z = rng.standard_t(float(degrees_of_freedom), size=simulations) * innovation_scale
        else:
            z = rng.standard_normal(simulations)

        variance_pct2 = np.maximum(variance_pct2, 1e-12)
        shocks_pct = z * np.sqrt(variance_pct2)
        cumulative_variance_pct2 += variance_pct2
        cumulative_log_returns += (mu + shocks_pct) / 100.0

        if fit.model == "GARCH":
            next_variance = (
                parameters["omega"]
                + parameters["alpha"] * shocks_pct**2
                + parameters["beta"] * variance_pct2
            )
        elif fit.model == "GJR-GARCH":
            downside = np.where(shocks_pct < 0.0, shocks_pct**2, 0.0)
            next_variance = (
                parameters["omega"]
                + parameters["alpha"] * shocks_pct**2
                + parameters["gamma"] * downside
                + parameters["beta"] * variance_pct2
            )
        else:
            log_variance = (
                parameters["omega"]
                + parameters["beta"] * np.log(variance_pct2)
                + parameters["alpha"] * (np.abs(z) - expected_abs)
                + parameters["gamma"] * z
            )
            next_variance = np.exp(np.clip(log_variance, -30.0, 30.0))

        if not np.all(np.isfinite(next_variance)) or np.any(next_variance <= 0.0):
            raise FloatingPointError("simulated variance path became invalid")
        variance_pct2 = next_variance

    prices = spot_price * np.exp(np.clip(cumulative_log_returns, -50.0, 50.0))
    if not np.all(np.isfinite(prices)):
        raise FloatingPointError("simulated prices are not finite")

    tail = (1.0 - confidence) / 2.0
    lower, median, upper = np.quantile(
        prices, [tail, 0.5, 1.0 - tail]
    )
    # A few extreme paths can dominate the arithmetic mean under heavy-tailed
    # innovations (especially for EGARCH). The median path standard deviation
    # is a more stable "typical move" summary; price quantiles still retain tails.
    # A Student-t innovation has no finite moment-generating function. In
    # EGARCH, non-zero asymmetry terms enter an exponential variance recursion,
    # so the expected future variance is not finite under Student-t shocks.
    # Keep path quantiles and the median typical-move summary, but do not expose
    # the Monte Carlo sample mean as an expected variance for this combination.
    if fit.model == "EGARCH" and fit.distribution == "student_t" and (
        abs(parameters["alpha"]) > 1e-12 or abs(parameters["gamma"]) > 1e-12
    ):
        expected_cumulative_variance = None
    else:
        expected_cumulative_variance = float(np.mean(cumulative_variance_pct2) / 10_000.0)
        if not np.isfinite(expected_cumulative_variance) or expected_cumulative_variance <= 0.0:
            raise FloatingPointError("expected cumulative variance is invalid")
    horizon_volatility = float(
        np.median(np.sqrt(cumulative_variance_pct2 / 10_000.0))
    )
    annualized_volatility = horizon_volatility * float(
        np.sqrt(periods_per_year / horizon_steps)
    )

    return HorizonPriceInterval(
        spot_price=float(spot_price),
        confidence=float(confidence),
        lower_price=float(lower),
        median_price=float(median),
        upper_price=float(upper),
        horizon_steps=horizon_steps,
        simulations=simulations,
        expected_cumulative_variance=expected_cumulative_variance,
        conditional_volatility=horizon_volatility,
        annualized_volatility=annualized_volatility,
        mean_log_return=mu * horizon_steps / 100.0,
        degrees_of_freedom=degrees_of_freedom,
    )
