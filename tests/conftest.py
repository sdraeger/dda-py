"""Shared test fixtures for dda-py tests."""

import numpy as np
import pytest

from dda_py.results import STResult, CTResult, DEResult


@pytest.fixture
def mock_st_result():
    """Factory fixture for creating mock STResult objects."""

    def _make(n_ch=3, n_win=10, n_coeff=3, seed=42):
        rng = np.random.RandomState(seed)
        return STResult(
            coefficients=rng.randn(n_ch, n_win, n_coeff),
            errors=rng.rand(n_ch, n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            channel_labels=[f"ch{i}" for i in range(n_ch)],
            params={
                "sfreq": 256.0,
                "wl": 200,
                "ws": 100,
                "delays": [7, 10],
                "model": [1, 2, 10],
            },
        )

    return _make


@pytest.fixture
def mock_ct_result():
    """Factory fixture for creating mock CTResult objects."""

    def _make(n_pairs=3, n_win=10, n_coeff=3, seed=42):
        rng = np.random.RandomState(seed)
        all_pairs = [
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 3),
        ]
        pair_labels = [f"ch{i}-ch{j}" for i, j in all_pairs[:n_pairs]]
        return CTResult(
            coefficients=rng.randn(n_pairs, n_win, n_coeff),
            errors=rng.rand(n_pairs, n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            pair_labels=pair_labels,
            params={
                "sfreq": 256.0,
                "wl": 200,
                "ws": 100,
                "delays": [7, 10],
                "model": [1, 2, 10],
            },
        )

    return _make


@pytest.fixture
def mock_de_result():
    """Factory fixture for creating mock DEResult objects."""

    def _make(n_win=10, seed=42):
        rng = np.random.RandomState(seed)
        return DEResult(
            ergodicity=rng.rand(n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            params={"sfreq": 256.0, "wl": 200, "ws": 100},
        )

    return _make
