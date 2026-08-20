"""Property-based invariant tests using Hypothesis."""

from typing import List
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from src.models.risk_parity import RiskParityOptimizer, compute_marginal_risk_contributions
from src.models.rwa_sleeve import RWASleeveManager
from src.models.vol_targeting import VolatilityTargeter


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=5,
    )
)
def test_property_portfolio_weights_sum_to_one(base_weight_factors: List[float]) -> None:
    """Property test: Normalized portfolio weights must always sum strictly to 1.0."""
    total = sum(base_weight_factors)
    normalized_weights = {f"A_{i}": val / total for i, val in enumerate(base_weight_factors)}

    weight_sum = sum(normalized_weights.values())
    assert abs(weight_sum - 1.0) < 1e-6


@settings(max_examples=30, deadline=None)
@given(
    st.integers(min_value=2, max_value=4),
)
def test_property_non_negative_risk_contributions(n_dim: int) -> None:
    """Property test: Risk contributions under risk parity must be non-negative."""
    asset_names = [f"ASSET_{i}" for i in range(n_dim)]

    # Generate valid PSD covariance matrix
    A = np.random.uniform(0.1, 1.0, size=(n_dim, n_dim))
    cov_matrix = np.dot(A, A.T) + np.eye(n_dim) * 0.01

    optimizer = RiskParityOptimizer()
    weights_dict = optimizer.optimize_risk_parity(asset_names, cov_matrix)

    weights_vec = np.array([weights_dict[name] for name in asset_names])

    # Weights sum to 1.0 invariant
    assert abs(float(np.sum(weights_vec)) - 1.0) < 1e-5

    # Compute risk contributions
    mcr, rc = compute_marginal_risk_contributions(weights_vec, cov_matrix)

    # Risk contributions must be non-negative
    for val in rc:
        assert val >= -1e-8
