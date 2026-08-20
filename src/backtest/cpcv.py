"""Combinatorial Purged Cross-Validation (CPCV) Partitioning Engine."""

import itertools
from typing import Dict, List, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict


class CPCVSplit(BaseModel):
    model_config = ConfigDict(frozen=True)

    split_id: int
    train_indices: List[int]
    test_indices: List[int]


class CombinatorialPurgedCrossValidation:
    """CPCV Partitioning with Purging and Embargoing to eliminate lookahead and autocorrelation leakage."""

    def __init__(
        self,
        n_splits: int = 6,
        k_test_splits: int = 2,
        purge_window: int = 5,
        embargo_window: int = 5,
    ) -> None:
        self.n_splits = n_splits
        self.k_test_splits = k_test_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def generate_splits(self, n_samples: int) -> List[CPCVSplit]:
        """Generates all C(n_splits, k_test_splits) CPCV splits with purging and embargoing."""
        if n_samples <= 0:
            return []

        block_bounds = self._get_block_bounds(n_samples)
        combos = list(itertools.combinations(range(self.n_splits), self.k_test_splits))

        cpcv_splits: List[CPCVSplit] = []

        for split_id, test_blocks in enumerate(combos):
            test_indices_set: set[int] = set()
            for b_idx in test_blocks:
                start_i, end_i = block_bounds[b_idx]
                test_indices_set.update(range(start_i, end_i))

            test_indices = sorted(list(test_indices_set))

            # Apply purging and embargoing to train set
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # Purge samples prior to test blocks that overlap label windows
            # Embargo samples following test blocks to prevent autocorrelation
            for b_idx in test_blocks:
                start_i, end_i = block_bounds[b_idx]

                purge_start = max(0, start_i - self.purge_window)
                train_mask[purge_start:start_i] = False

                embargo_end = min(n_samples, end_i + self.embargo_window)
                train_mask[end_i:embargo_end] = False

            train_indices = np.where(train_mask)[0].tolist()

            cpcv_splits.append(
                CPCVSplit(
                    split_id=split_id,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )

        return cpcv_splits

    def _get_block_bounds(self, n_samples: int) -> List[Tuple[int, int]]:
        """Splits n_samples into n_splits contiguous blocks."""
        block_size = n_samples // self.n_splits
        bounds = []
        for i in range(self.n_splits):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_splits - 1 else n_samples
            bounds.append((start, end))
        return bounds
