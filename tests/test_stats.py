"""Tests for statistics module."""

import numpy as np
import pytest

from dda_py.stats import (
    EffectSizeResult,
    PermutationResult,
    WindowComparisonResult,
    _extract_coefficients,
    compute_effect_size,
    permutation_test,
)


class TestExtractCoefficients:
    def test_consistent_shapes(self, mock_st_result):
        results = [mock_st_result(seed=i) for i in range(5)]
        data = _extract_coefficients(results)
        assert data.shape == (5, 3, 3)  # (n_subjects, n_channels, n_coeffs)

    def test_mismatched_channels_raises(self, mock_st_result):
        results = [mock_st_result(n_ch=3), mock_st_result(n_ch=4, seed=1)]
        with pytest.raises(ValueError, match="Channel count"):
            _extract_coefficients(results)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _extract_coefficients([])


class TestPermutationTest:
    def test_returns_permutation_result(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(10)]
        group_b = [mock_st_result(seed=i + 100) for i in range(10)]
        result = permutation_test(group_a, group_b, n_permutations=100, seed=42)

        assert isinstance(result, PermutationResult)
        assert result.observed_stat.shape == (3, 3)
        assert result.p_value.shape == (3, 3)
        assert result.null_distribution.shape == (100, 3, 3)
        assert result.n_permutations == 100

    def test_p_value_range(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(10)]
        group_b = [mock_st_result(seed=i + 100) for i in range(10)]
        result = permutation_test(group_a, group_b, n_permutations=100, seed=42)

        assert np.all(result.p_value >= 0)
        assert np.all(result.p_value <= 1)

    def test_seed_reproducibility(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]

        r1 = permutation_test(group_a, group_b, n_permutations=100, seed=42)
        r2 = permutation_test(group_a, group_b, n_permutations=100, seed=42)

        np.testing.assert_array_equal(r1.p_value, r2.p_value)
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_identical_groups_high_pvalue(self, mock_st_result):
        group = [mock_st_result(seed=i) for i in range(10)]
        result = permutation_test(group, group, n_permutations=500, seed=42)

        # Identical groups → no real difference → p-values should be high
        assert np.all(result.p_value > 0.05)

    def test_shifted_groups_low_pvalue(self, mock_st_result):
        from dda_py.results import STResult

        group_a = [mock_st_result(seed=i) for i in range(20)]
        # Create group_b with coefficients shifted by 5.0
        group_b = []
        for i in range(20):
            r = mock_st_result(seed=i + 100)
            shifted = STResult(
                coefficients=r.coefficients + 5.0,
                errors=r.errors,
                window_starts=r.window_starts,
                window_ends=r.window_ends,
                channel_labels=r.channel_labels,
                params=r.params,
            )
            group_b.append(shifted)

        result = permutation_test(group_a, group_b, n_permutations=500, seed=42)
        # Large shift → should detect significant difference
        assert np.any(result.p_value < 0.05)

    def test_tail_options(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]

        for tail in [-1, 0, 1]:
            result = permutation_test(
                group_a, group_b, n_permutations=50, tail=tail, seed=42
            )
            assert result.tail == tail

    def test_invalid_tail_raises(self, mock_st_result):
        group = [mock_st_result()]
        with pytest.raises(ValueError, match="tail"):
            permutation_test(group, group, n_permutations=10, tail=2)

    def test_custom_stat_fun(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]

        def max_diff(a, b):
            return a.max(axis=0) - b.max(axis=0)

        result = permutation_test(
            group_a, group_b, n_permutations=50, stat_fun=max_diff, seed=42
        )
        assert result.observed_stat.shape == (3, 3)

    def test_to_dataframe(self, mock_st_result):
        pd = pytest.importorskip("pandas")
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]
        result = permutation_test(group_a, group_b, n_permutations=50, seed=42)
        df = result.to_dataframe()
        assert len(df) == 3 * 3  # n_channels * n_coeffs
        assert "p_value" in df.columns


class TestComputeEffectSize:
    def test_returns_effect_size(self, mock_st_result):
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]
        result = compute_effect_size(group_a, group_b)

        assert isinstance(result, EffectSizeResult)
        assert result.cohens_d.shape == (3, 3)
        assert len(result.channel_labels) == 3

    def test_identical_groups_zero_d(self, mock_st_result):
        group = [mock_st_result(seed=i) for i in range(10)]
        result = compute_effect_size(group, group)
        np.testing.assert_allclose(result.cohens_d, 0.0, atol=1e-10)

    def test_to_dataframe(self, mock_st_result):
        pd = pytest.importorskip("pandas")
        group_a = [mock_st_result(seed=i) for i in range(5)]
        group_b = [mock_st_result(seed=i + 50) for i in range(5)]
        result = compute_effect_size(group_a, group_b)
        df = result.to_dataframe()
        assert len(df) == 3 * 3
        assert "cohens_d" in df.columns


class TestCompareWindows:
    def test_returns_comparison_result(self, mock_st_result):
        pytest.importorskip("scipy")
        from dda_py.stats import compare_windows

        result = mock_st_result(n_win=20, seed=42)
        comp = compare_windows(result, baseline_windows=slice(0, 10), test_windows=slice(10, 20))

        assert isinstance(comp, WindowComparisonResult)
        assert comp.stat.shape == (3, 3)
        assert comp.p_value.shape == (3, 3)
        assert comp.baseline_mean.shape == (3, 3)
        assert comp.test_mean.shape == (3, 3)

    def test_with_list_indices(self, mock_st_result):
        pytest.importorskip("scipy")
        from dda_py.stats import compare_windows

        result = mock_st_result(n_win=20, seed=42)
        comp = compare_windows(
            result,
            baseline_windows=[0, 1, 2, 3, 4],
            test_windows=[15, 16, 17, 18, 19],
        )
        assert comp.stat.shape == (3, 3)

    def test_ranksum_method(self, mock_st_result):
        pytest.importorskip("scipy")
        from dda_py.stats import compare_windows

        result = mock_st_result(n_win=20, seed=42)
        comp = compare_windows(
            result,
            baseline_windows=slice(0, 10),
            test_windows=slice(10, 20),
            method="ranksum",
        )
        assert comp.stat.shape == (3, 3)
