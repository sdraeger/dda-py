"""Tests for result dataclasses."""

import numpy as np
import pytest

from dda_py.results import STResult, CTResult, DEResult


class TestSTResult:
    """Test STResult dataclass."""

    def _make_result(self, n_ch=3, n_win=10, n_coeff=3):
        return STResult(
            coefficients=np.random.randn(n_ch, n_win, n_coeff),
            errors=np.random.rand(n_ch, n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            channel_labels=[f"ch{i}" for i in range(n_ch)],
            params={"sfreq": 256.0, "wl": 200, "ws": 100},
        )

    def test_shapes(self):
        r = self._make_result(3, 10, 3)
        assert r.coefficients.shape == (3, 10, 3)
        assert r.errors.shape == (3, 10)
        assert r.window_starts.shape == (10,)
        assert r.window_ends.shape == (10,)

    def test_properties(self):
        r = self._make_result(4, 20, 5)
        assert r.n_channels == 4
        assert r.n_windows == 20
        assert r.n_coeffs == 5

    def test_to_dataframe(self):
        r = self._make_result(2, 5, 3)
        df = r.to_dataframe()

        assert len(df) == 2 * 5  # n_channels * n_windows
        assert "channel" in df.columns
        assert "window_start" in df.columns
        assert "window_end" in df.columns
        assert "a_1" in df.columns
        assert "a_2" in df.columns
        assert "a_3" in df.columns
        assert "error" in df.columns
        assert set(df["channel"].unique()) == {"ch0", "ch1"}

    def test_empty_result(self):
        r = STResult(
            coefficients=np.empty((2, 0, 0)),
            errors=np.empty((2, 0)),
            window_starts=np.empty(0, dtype=np.int64),
            window_ends=np.empty(0, dtype=np.int64),
            channel_labels=["ch0", "ch1"],
            params={},
        )
        assert r.n_channels == 2
        assert r.n_windows == 0


class TestCTResult:
    """Test CTResult dataclass."""

    def _make_result(self, n_pairs=3, n_win=10, n_coeff=3):
        return CTResult(
            coefficients=np.random.randn(n_pairs, n_win, n_coeff),
            errors=np.random.rand(n_pairs, n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            pair_labels=[f"ch{i}-ch{j}" for i, j in [(0, 1), (0, 2), (1, 2)][:n_pairs]],
            params={"sfreq": 256.0},
        )

    def test_shapes(self):
        r = self._make_result(3, 10, 3)
        assert r.coefficients.shape == (3, 10, 3)
        assert r.errors.shape == (3, 10)

    def test_properties(self):
        r = self._make_result(3, 15, 4)
        assert r.n_pairs == 3
        assert r.n_windows == 15
        assert r.n_coeffs == 4

    def test_to_dataframe(self):
        r = self._make_result(2, 5, 3)
        df = r.to_dataframe()

        assert len(df) == 2 * 5
        assert "pair" in df.columns
        assert "a_1" in df.columns
        assert "error" in df.columns


class TestDEResult:
    """Test DEResult dataclass."""

    def _make_result(self, n_win=10):
        return DEResult(
            ergodicity=np.random.rand(n_win),
            window_starts=np.arange(0, n_win * 100, 100, dtype=np.int64),
            window_ends=np.arange(200, n_win * 100 + 200, 100, dtype=np.int64),
            params={"sfreq": 256.0},
        )

    def test_shapes(self):
        r = self._make_result(10)
        assert r.ergodicity.shape == (10,)
        assert r.window_starts.shape == (10,)

    def test_properties(self):
        r = self._make_result(20)
        assert r.n_windows == 20

    def test_to_dataframe(self):
        r = self._make_result(5)
        df = r.to_dataframe()

        assert len(df) == 5
        assert "ergodicity" in df.columns
        assert "window_start" in df.columns
        assert "window_end" in df.columns

    def test_empty_result(self):
        r = DEResult(
            ergodicity=np.empty(0),
            window_starts=np.empty(0, dtype=np.int64),
            window_ends=np.empty(0, dtype=np.int64),
            params={},
        )
        assert r.n_windows == 0
