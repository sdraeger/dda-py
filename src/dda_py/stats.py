"""Statistical testing for DDA results."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .results import CTResult, STResult


@dataclass
class PermutationResult:
    """Result of a permutation test.

    Attributes:
        observed_stat: Observed test statistic, shape (n_channels, n_coeffs).
        p_value: p-value per channel per coefficient, shape (n_channels, n_coeffs).
        null_distribution: Null distribution, shape (n_permutations, n_channels, n_coeffs).
        n_permutations: Number of permutations performed.
        tail: 0=two-sided, 1=greater, -1=less.
    """

    observed_stat: np.ndarray
    p_value: np.ndarray
    null_distribution: np.ndarray
    n_permutations: int
    tail: int

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to DataFrame with channel, coefficient, observed_stat, p_value."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install 'dda-py[pandas]'"
            )
        rows: list[dict[str, Any]] = []
        n_ch, n_coeff = self.observed_stat.shape
        for ch in range(n_ch):
            for c in range(n_coeff):
                rows.append(
                    {
                        "channel": ch,
                        "coefficient": f"a_{c + 1}",
                        "observed_stat": float(self.observed_stat[ch, c]),
                        "p_value": float(self.p_value[ch, c]),
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class EffectSizeResult:
    """Cohen's d effect sizes.

    Attributes:
        cohens_d: Effect size, shape (n_channels, n_coeffs).
        channel_labels: Channel or pair labels.
    """

    cohens_d: np.ndarray
    channel_labels: List[str]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install 'dda-py[pandas]'"
            )
        rows: list[dict[str, Any]] = []
        n_ch, n_coeff = self.cohens_d.shape
        for ch_idx, label in enumerate(self.channel_labels):
            for c in range(n_coeff):
                rows.append(
                    {
                        "channel": label,
                        "coefficient": f"a_{c + 1}",
                        "cohens_d": float(self.cohens_d[ch_idx, c]),
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class WindowComparisonResult:
    """Result of within-subject window comparison.

    Attributes:
        stat: Test statistic, shape (n_channels, n_coeffs).
        p_value: p-value, shape (n_channels, n_coeffs).
        baseline_mean: Mean coefficients in baseline, shape (n_channels, n_coeffs).
        test_mean: Mean coefficients in test, shape (n_channels, n_coeffs).
    """

    stat: np.ndarray
    p_value: np.ndarray
    baseline_mean: np.ndarray
    test_mean: np.ndarray

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install 'dda-py[pandas]'"
            )
        rows: list[dict[str, Any]] = []
        n_ch, n_coeff = self.stat.shape
        for ch in range(n_ch):
            for c in range(n_coeff):
                rows.append(
                    {
                        "channel": ch,
                        "coefficient": f"a_{c + 1}",
                        "stat": float(self.stat[ch, c]),
                        "p_value": float(self.p_value[ch, c]),
                        "baseline_mean": float(self.baseline_mean[ch, c]),
                        "test_mean": float(self.test_mean[ch, c]),
                    }
                )
        return pd.DataFrame(rows)


def _extract_coefficients(
    results: "List[Union[STResult, CTResult]]",
) -> np.ndarray:
    """Extract and stack window-averaged coefficients.

    Returns:
        Array of shape (n_subjects, n_channels, n_coeffs).
    """
    from .results import CTResult, STResult

    if not results:
        raise ValueError("Cannot extract from empty results list")

    first = results[0]
    n_ch = first.n_channels if isinstance(first, STResult) else first.n_pairs
    n_coeff = first.n_coeffs

    for i, r in enumerate(results):
        r_ch = r.n_channels if isinstance(r, STResult) else r.n_pairs
        if r_ch != n_ch:
            raise ValueError(
                f"Channel count mismatch: expected {n_ch}, got {r_ch} at index {i}"
            )
        if r.n_coeffs != n_coeff:
            raise ValueError(
                f"Coefficient count mismatch: expected {n_coeff}, "
                f"got {r.n_coeffs} at index {i}"
            )

    # Average across windows to get per-subject means
    means = np.stack(
        [r.coefficients.mean(axis=1) for r in results], axis=0
    )  # (n_subjects, n_ch, n_coeff)
    return means


def _default_stat_fun(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Difference of means along axis 0."""
    return a.mean(axis=0) - b.mean(axis=0)


def permutation_test(
    group_a: "List[Union[STResult, CTResult]]",
    group_b: "List[Union[STResult, CTResult]]",
    n_permutations: int = 10000,
    stat_fun: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    tail: int = 0,
    seed: Optional[int] = None,
) -> PermutationResult:
    """Non-parametric permutation test comparing two groups of DDA results.

    Averages coefficients across windows per subject, then tests
    whether the groups differ.

    Args:
        group_a: List of STResult or CTResult for group A.
        group_b: List of STResult or CTResult for group B.
        n_permutations: Number of random permutations.
        stat_fun: Function(a, b) -> stat. Takes arrays (n_subjects, n_ch, n_coeff).
            Default: difference of means.
        tail: 0=two-sided, 1=greater (A > B), -1=less (A < B).
        seed: Random seed for reproducibility.

    Returns:
        PermutationResult.
    """
    if stat_fun is None:
        stat_fun = _default_stat_fun

    data_a = _extract_coefficients(group_a)  # (n_a, n_ch, n_coeff)
    data_b = _extract_coefficients(group_b)  # (n_b, n_ch, n_coeff)

    n_a = data_a.shape[0]
    combined = np.concatenate([data_a, data_b], axis=0)  # (n_total, n_ch, n_coeff)
    n_total = combined.shape[0]

    observed = stat_fun(data_a, data_b)  # (n_ch, n_coeff)

    rng = np.random.RandomState(seed)
    null_dist = np.empty(
        (n_permutations, *observed.shape), dtype=np.float64
    )

    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_a = combined[perm[:n_a]]
        perm_b = combined[perm[n_a:]]
        null_dist[i] = stat_fun(perm_a, perm_b)

    # Compute p-values
    if tail == 0:
        p_value = (
            np.sum(np.abs(null_dist) >= np.abs(observed), axis=0) + 1
        ) / (n_permutations + 1)
    elif tail == 1:
        p_value = (np.sum(null_dist >= observed, axis=0) + 1) / (
            n_permutations + 1
        )
    elif tail == -1:
        p_value = (np.sum(null_dist <= observed, axis=0) + 1) / (
            n_permutations + 1
        )
    else:
        raise ValueError(f"tail must be -1, 0, or 1, got {tail}")

    return PermutationResult(
        observed_stat=observed,
        p_value=p_value,
        null_distribution=null_dist,
        n_permutations=n_permutations,
        tail=tail,
    )


def compare_windows(
    result: "Union[STResult, CTResult]",
    baseline_windows: Union[List[int], slice],
    test_windows: Union[List[int], slice],
    method: str = "ttest",
) -> WindowComparisonResult:
    """Within-subject comparison of baseline vs test time windows.

    Args:
        result: Single subject result.
        baseline_windows: Window indices (or slice) for baseline.
        test_windows: Window indices (or slice) for test condition.
        method: 'ttest' or 'ranksum'.

    Returns:
        WindowComparisonResult.
    """
    try:
        from scipy.stats import ranksums, ttest_ind
    except ImportError:
        raise ImportError(
            "scipy is required for compare_windows(). "
            "Install with: pip install 'dda-py[scipy]'"
        )

    coeffs = result.coefficients  # (n_ch, n_win, n_coeff)

    if isinstance(baseline_windows, slice):
        baseline_data = coeffs[:, baseline_windows, :]
    else:
        baseline_data = coeffs[:, baseline_windows, :]

    if isinstance(test_windows, slice):
        test_data = coeffs[:, test_windows, :]
    else:
        test_data = coeffs[:, test_windows, :]

    n_ch, _, n_coeff = coeffs.shape
    stat_arr = np.empty((n_ch, n_coeff), dtype=np.float64)
    p_arr = np.empty((n_ch, n_coeff), dtype=np.float64)

    test_func = ttest_ind if method == "ttest" else ranksums

    for ch in range(n_ch):
        for c in range(n_coeff):
            s, p = test_func(baseline_data[ch, :, c], test_data[ch, :, c])
            stat_arr[ch, c] = s
            p_arr[ch, c] = p

    return WindowComparisonResult(
        stat=stat_arr,
        p_value=p_arr,
        baseline_mean=baseline_data.mean(axis=1),
        test_mean=test_data.mean(axis=1),
    )


def compute_effect_size(
    group_a: "List[Union[STResult, CTResult]]",
    group_b: "List[Union[STResult, CTResult]]",
) -> EffectSizeResult:
    """Compute Cohen's d per coefficient per channel.

    Averages across windows per subject, then computes Cohen's d.

    Returns:
        EffectSizeResult.
    """
    from .results import STResult

    data_a = _extract_coefficients(group_a)  # (n_a, n_ch, n_coeff)
    data_b = _extract_coefficients(group_b)

    mean_a = data_a.mean(axis=0)
    mean_b = data_b.mean(axis=0)

    n_a = data_a.shape[0]
    n_b = data_b.shape[0]

    var_a = data_a.var(axis=0, ddof=1)
    var_b = data_b.var(axis=0, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    # Avoid division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.where(pooled_std > 0, (mean_a - mean_b) / pooled_std, 0.0)

    first = group_a[0]
    if isinstance(first, STResult):
        channel_labels = first.channel_labels
    else:
        channel_labels = first.pair_labels

    return EffectSizeResult(cohens_d=d, channel_labels=channel_labels)
