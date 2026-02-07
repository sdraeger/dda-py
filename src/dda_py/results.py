"""Structured result types for DDA analysis output.

Each result class wraps numpy arrays with metadata and provides
convenience methods for conversion to pandas DataFrames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class STResult:
    """Single Timeseries DDA result.

    Attributes:
        coefficients: Coefficient values, shape (n_channels, n_windows, n_coeffs).
        errors: Residual error per channel per window, shape (n_channels, n_windows).
        window_starts: Window start sample indices, shape (n_windows,).
        window_ends: Window end sample indices, shape (n_windows,).
        channel_labels: Channel name strings.
        params: Parameters used for this analysis run.
    """
    coefficients: np.ndarray
    errors: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    channel_labels: List[str]
    params: Dict[str, Any]

    @property
    def n_channels(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_windows(self) -> int:
        return self.coefficients.shape[1]

    @property
    def n_coeffs(self) -> int:
        return self.coefficients.shape[2]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to long-format pandas DataFrame.

        Returns:
            DataFrame with columns: channel, window_start, window_end,
            a_1, a_2, ..., a_N, error.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        rows = []
        for ch_idx, label in enumerate(self.channel_labels):
            for win_idx in range(self.n_windows):
                row: Dict[str, Any] = {
                    "channel": label,
                    "window_start": int(self.window_starts[win_idx]),
                    "window_end": int(self.window_ends[win_idx]),
                }
                for c in range(self.n_coeffs):
                    row[f"a_{c + 1}"] = float(self.coefficients[ch_idx, win_idx, c])
                row["error"] = float(self.errors[ch_idx, win_idx])
                rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class CTResult:
    """Cross-Timeseries DDA result.

    Attributes:
        coefficients: Coefficient values, shape (n_pairs, n_windows, n_coeffs).
        errors: Residual error per pair per window, shape (n_pairs, n_windows).
        window_starts: Window start sample indices, shape (n_windows,).
        window_ends: Window end sample indices, shape (n_windows,).
        pair_labels: Pair label strings (e.g. ["ch0-ch1", "ch0-ch2"]).
        params: Parameters used for this analysis run.
    """
    coefficients: np.ndarray
    errors: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    pair_labels: List[str]
    params: Dict[str, Any]

    @property
    def n_pairs(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_windows(self) -> int:
        return self.coefficients.shape[1]

    @property
    def n_coeffs(self) -> int:
        return self.coefficients.shape[2]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to long-format pandas DataFrame.

        Returns:
            DataFrame with columns: pair, window_start, window_end,
            a_1, a_2, ..., a_N, error.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        rows = []
        for pair_idx, label in enumerate(self.pair_labels):
            for win_idx in range(self.n_windows):
                row: Dict[str, Any] = {
                    "pair": label,
                    "window_start": int(self.window_starts[win_idx]),
                    "window_end": int(self.window_ends[win_idx]),
                }
                for c in range(self.n_coeffs):
                    row[f"a_{c + 1}"] = float(self.coefficients[pair_idx, win_idx, c])
                row["error"] = float(self.errors[pair_idx, win_idx])
                rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class DEResult:
    """Delay Embedding (Dynamical Ergodicity) DDA result.

    Attributes:
        ergodicity: Ergodicity values per window, shape (n_windows,).
        window_starts: Window start sample indices, shape (n_windows,).
        window_ends: Window end sample indices, shape (n_windows,).
        params: Parameters used for this analysis run.
    """
    ergodicity: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    params: Dict[str, Any]

    @property
    def n_windows(self) -> int:
        return self.ergodicity.shape[0]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with columns: window_start, window_end, ergodicity.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        return pd.DataFrame({
            "window_start": self.window_starts.astype(int),
            "window_end": self.window_ends.astype(int),
            "ergodicity": self.ergodicity,
        })
