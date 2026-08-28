"""Risk Parity Convex Optimization Engine (CVXPY & SciPy)."""

from typing import Any, Dict, List, Tuple
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


def compute_marginal_risk_contributions(
    weights: np.ndarray[Any, Any], cov_matrix: np.ndarray[Any, Any]
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Computes MCR and RC vectors for a given weight vector and covariance matrix.

    MCR_i = (Sigma w)_i / sqrt(w^T Sigma w)
    RC_i  = w_i * MCR_i
    """
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if portfolio_vol < 1e-12:
        mcr = np.zeros_like(weights)
        rc = np.zeros_like(weights)
        return mcr, rc

    mcr = np.dot(cov_matrix, weights) / portfolio_vol
    rc = weights * mcr
    return mcr, rc


class RiskParityOptimizer:
    """Convex Risk Parity Optimizer to equalize Marginal Risk Contributions."""

    def __init__(self, tol: float = 1e-8) -> None:
        self.tol = tol

    def optimize_risk_parity(
        self, asset_names: List[str], cov_matrix: np.ndarray[Any, Any]
    ) -> Dict[str, float]:
        """Equalizes Marginal Risk Contribution (MRC) across components."""
        n = len(asset_names)
        if n == 0:
            return {}
        if n == 1:
            return {asset_names[0]: 1.0}

        # Check positive semi-definiteness
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
        cov_matrix += np.eye(n) * 1e-8  # Regularization for numerical stability

        weights = self._solve_cvxpy(n, cov_matrix)
        if weights is None:
            weights = self._solve_scipy(n, cov_matrix)

        # Normalize weights
        weights = np.maximum(weights, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(n) / n

        return {name: float(weights[i]) for i, name in enumerate(asset_names)}

    def _solve_cvxpy(self, n: int, cov_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
        """Solves log-barrier Risk Parity formulation using CVXPY:

        min 0.5 * y^T Sigma y - sum(log(y_i))
        w = y / sum(y)
        """
        try:
            y = cp.Variable(n, pos=True)
            quad_form = cp.quad_form(y, cov_matrix)  # type: ignore[attr-defined]
            log_sum = cp.sum(cp.log(y))  # type: ignore[attr-defined]
            objective = cp.Minimize(0.5 * quad_form - log_sum)
            problem = cp.Problem(objective)
            problem.solve(solver=cp.CLARABEL, verbose=False)  # type: ignore[no-untyped-call]

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and y.value is not None:
                y_val: np.ndarray[Any, Any] = np.array(y.value).flatten()
                res: np.ndarray[Any, Any] = y_val / float(np.sum(y_val))
                return res
        except Exception:
            pass
        return None

    def _solve_scipy(self, n: int, cov_matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fallback SciPy SLSQP optimizer directly minimizing sum of squared RC differences."""

        def objective(w: np.ndarray[Any, Any]) -> float:
            mcr, rc = compute_marginal_risk_contributions(w, cov_matrix)
            diffs = rc[:, None] - rc[None, :]
            return float(np.sum(diffs**2))

        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        x0 = np.ones(n) / n

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
        )

        if res.success and res.x is not None:
            ans: np.ndarray[Any, Any] = np.array(res.x)
            return ans
        return x0
