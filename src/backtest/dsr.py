"""Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and Overfitting Gate."""

import math
from typing import List, Sequence
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import kurtosis, norm, skew  # type: ignore[import-untyped]


class StrategyOverfittedException(Exception):
    """Raised when strategy fails PBO (<0.10) or DSR (>=0.95) overfitting verification gates."""

    pass


class DSRResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dsr_score: float  # Value between 0 and 1 (p-value equivalent)
    pbo_score: float  # Probability of Backtest Overfitting (0 to 1)
    estimated_sharpe: float
    benchmark_sharpe_star: float
    skewness: float
    kurtosis: float
    num_trials: int
    num_observations: int
    is_valid: bool


class DeflatedSharpeRatioCalculator:
    """Calculates Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO)."""

    EULER_MASCHERONI = 0.57721566490153286

    @classmethod
    def calculate_benchmark_sharpe_star(
        cls, trial_sharpes: Sequence[float], num_trials: int
    ) -> float:
        """Calculates expected maximum Sharpe ratio SR* under null hypothesis of multiple testing."""
        if num_trials <= 1:
            return 0.0

        var_sr = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
        if var_sr <= 1e-12:
            return 0.0

        q1 = 1.0 - (1.0 / num_trials)
        q2 = 1.0 - (1.0 / (num_trials * math.e))

        z1 = float(norm.ppf(max(1e-6, min(0.999999, q1))))
        z2 = float(norm.ppf(max(1e-6, min(0.999999, q2))))

        sr_star = float(np.sqrt(var_sr)) * (
            (1.0 - cls.EULER_MASCHERONI) * z1 + cls.EULER_MASCHERONI * z2
        )
        return sr_star

    @classmethod
    def calculate_dsr(
        cls,
        returns: Sequence[float],
        all_trial_sharpes: Sequence[float],
        annualization_factor: float = 365.0,
    ) -> DSRResult:
        """Calculates Deflated Sharpe Ratio correcting for skewness, kurtosis, and multiple testing."""
        arr = np.array(returns, dtype=np.float64)
        t_len = len(arr)
        if t_len < 4:
            return DSRResult(
                dsr_score=0.0,
                pbo_score=1.0,
                estimated_sharpe=0.0,
                benchmark_sharpe_star=0.0,
                skewness=0.0,
                kurtosis=3.0,
                num_trials=len(all_trial_sharpes),
                num_observations=t_len,
                is_valid=False,
            )

        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        if std_ret <= 1e-12:
            sr_est = 0.0
        else:
            sr_est = (mean_ret / std_ret) * float(np.sqrt(annualization_factor))

        s_skew = float(skew(arr))
        k_kurt = float(kurtosis(arr, fisher=False))  # Excess kurtosis false -> normal kurtosis=3

        num_trials = max(1, len(all_trial_sharpes))
        sr_star = cls.calculate_benchmark_sharpe_star(all_trial_sharpes, num_trials)

        # Standard error denominator under non-Gaussianity
        denom = 1.0 - s_skew * sr_est + ((k_kurt - 1.0) / 4.0) * (sr_est**2)
        if denom <= 1e-12:
            denom = 1e-12

        se = float(np.sqrt(denom / max(1, t_len - 1)))
        z_stat = (sr_est - sr_star) / se
        dsr_score = float(norm.cdf(z_stat))

        # Calculate PBO as proportion of trial sharpes below 0 or below sr_star
        pbo_score = float(np.mean([1.0 if sr <= 0 else 0.0 for sr in all_trial_sharpes])) if all_trial_sharpes else 0.0

        is_valid = dsr_score >= 0.95 and pbo_score < 0.10

        return DSRResult(
            dsr_score=dsr_score,
            pbo_score=pbo_score,
            estimated_sharpe=sr_est,
            benchmark_sharpe_star=sr_star,
            skewness=s_skew,
            kurtosis=k_kurt,
            num_trials=num_trials,
            num_observations=t_len,
            is_valid=is_valid,
        )

    @classmethod
    def validate_overfitting_gate(cls, dsr_result: DSRResult) -> None:
        """Enforces out-of-sample verification gate (PBO < 0.10 and DSR >= 0.95).

        Raises StrategyOverfittedException if violated.
        """
        if dsr_result.pbo_score >= 0.10 or dsr_result.dsr_score < 0.95:
            raise StrategyOverfittedException(
                f"Strategy failed overfitting verification gate: "
                f"PBO={dsr_result.pbo_score:.4f} (must be < 0.10), "
                f"DSR={dsr_result.dsr_score:.4f} (must be >= 0.95)."
            )
