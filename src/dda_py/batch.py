"""Batch processing for DDA analysis over multiple files."""

from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .results import CTResult, DEResult, STResult


@dataclass
class GroupResult:
    """Stacked results from multiple subjects.

    Attributes:
        coefficients: Shape (n_subjects, n_channels, n_windows, n_coeffs).
        errors: Shape (n_subjects, n_channels, n_windows).
        subject_labels: Subject identifiers.
        channel_labels: Channel or pair labels (from first result).
        params: Parameters dict (from first result).
        variant: 'ST', 'CT', or 'DE'.
    """

    coefficients: np.ndarray
    errors: np.ndarray
    subject_labels: List[str]
    channel_labels: List[str]
    params: Dict[str, Any]
    variant: str

    @property
    def n_subjects(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_channels(self) -> int:
        return self.coefficients.shape[1]

    @property
    def n_windows(self) -> int:
        return self.coefficients.shape[2]

    @property
    def n_coeffs(self) -> int:
        return self.coefficients.shape[3]

    def mean_over_windows(self) -> np.ndarray:
        """Average coefficients across windows.

        Returns:
            Array of shape (n_subjects, n_channels, n_coeffs).
        """
        return self.coefficients.mean(axis=2)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to long-format DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install 'dda-py[pandas]'"
            )
        rows: list[dict[str, Any]] = []
        for subj_idx, subj_label in enumerate(self.subject_labels):
            for ch_idx, ch_label in enumerate(self.channel_labels):
                for win_idx in range(self.n_windows):
                    row: Dict[str, Any] = {
                        "subject": subj_label,
                        "channel": ch_label,
                        "window": win_idx,
                    }
                    for c in range(self.n_coeffs):
                        row[f"a_{c + 1}"] = float(
                            self.coefficients[subj_idx, ch_idx, win_idx, c]
                        )
                    row["error"] = float(self.errors[subj_idx, ch_idx, win_idx])
                    rows.append(row)
        return pd.DataFrame(rows)


def _get_run_func(variant: str):
    """Get the run function for a variant."""
    from .api import run_ct, run_de, run_st

    dispatch = {"st": run_st, "ct": run_ct, "de": run_de}
    key = variant.lower()
    if key not in dispatch:
        raise ValueError(
            f"Unknown variant '{variant}'. Must be one of: st, ct, de"
        )
    return dispatch[key]


def _process_file(args: tuple) -> Any:
    """Process a single file. Top-level function for pickling."""
    file_path, variant, kwargs = args
    run_func = _get_run_func(variant)
    data = np.loadtxt(file_path).T  # (n_samples, n_channels) -> (n_channels, n_samples)
    return run_func(data, **kwargs)


def run_batch(
    files: List[str],
    variant: str = "st",
    sfreq: float = 1.0,
    delays: Sequence[int] = (7, 10),
    model: Optional[List[int]] = None,
    wl: int = 200,
    ws: int = 100,
    channels: Optional[Union[List[int], List[str]]] = None,
    binary_path: Optional[str] = None,
    n_jobs: int = 1,
    progress: bool = True,
    labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> "List[Union[STResult, CTResult, DEResult]]":
    """Run DDA analysis on multiple files in batch.

    Args:
        files: List of file paths (ASCII text format, whitespace-delimited).
        variant: 'st', 'ct', or 'de' (case-insensitive).
        sfreq: Sampling frequency.
        delays: Delay values.
        model: Model encoding.
        wl: Window length.
        ws: Window step.
        channels: Channel selection.
        binary_path: Path to DDA binary.
        n_jobs: Number of parallel workers. 1 = sequential.
        progress: Show progress bar (uses tqdm if available).
        labels: Subject labels corresponding to files.
        **kwargs: Extra keyword args passed to run_st/run_ct/run_de.

    Returns:
        List of result objects, one per file.
    """
    # Validate variant early
    _get_run_func(variant)

    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Input file not found: {f}")

    run_kwargs: Dict[str, Any] = {
        "sfreq": sfreq,
        "delays": delays,
        "wl": wl,
        "ws": ws,
        **kwargs,
    }
    if model is not None:
        run_kwargs["model"] = model
    if channels is not None:
        run_kwargs["channels"] = channels
    if binary_path is not None:
        run_kwargs["binary_path"] = binary_path

    results: list[Any] = []

    if n_jobs == 1:
        run_func = _get_run_func(variant)
        iterator = enumerate(files)

        try:
            from tqdm import tqdm

            if progress:
                iterator = tqdm(
                    list(iterator), desc="DDA batch", unit="file"
                )
        except ImportError:
            pass

        for idx, file_path in iterator:
            if progress and "tqdm" not in sys.modules:
                basename = Path(file_path).name
                print(
                    f"Processing {idx + 1}/{len(files)}: {basename}",
                    file=sys.stderr,
                )
            data = np.loadtxt(file_path).T
            result = run_func(data, **run_kwargs)
            results.append(result)
    else:
        task_args = [
            (f, variant, run_kwargs) for f in files
        ]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_process_file, args): i
                for i, args in enumerate(task_args)
            }
            results = [None] * len(files)
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress:
                    basename = Path(files[idx]).name
                    print(
                        f"Completed {completed}/{len(files)}: {basename}",
                        file=sys.stderr,
                    )

    return results


def collect_results(
    results: "List[Union[STResult, CTResult, DEResult]]",
    labels: Optional[List[str]] = None,
) -> GroupResult:
    """Stack multiple results into a single GroupResult.

    All results must have matching channel/coefficient dimensions.
    Windows are truncated to the minimum across subjects.

    Args:
        results: List of result objects (all same type).
        labels: Subject labels. If None, uses 'subject_0', etc.

    Returns:
        GroupResult with 4D coefficient array.
    """
    from .results import CTResult, DEResult, STResult

    if not results:
        raise ValueError("Cannot collect empty results list")

    first = results[0]
    result_type = type(first)

    for i, r in enumerate(results):
        if type(r) is not result_type:
            raise ValueError(
                f"All results must be the same type. "
                f"Got {result_type.__name__} and {type(r).__name__} at index {i}"
            )

    if labels is None:
        labels = [f"subject_{i}" for i in range(len(results))]

    if isinstance(first, DEResult):
        min_win = min(r.n_windows for r in results)
        if any(r.n_windows != min_win for r in results):
            warnings.warn(
                f"Window counts vary across subjects. "
                f"Truncating to minimum: {min_win}",
                stacklevel=2,
            )

        coefficients = np.stack(
            [r.ergodicity[:min_win].reshape(1, min_win, 1) for r in results],
            axis=0,
        )
        errors = np.zeros(
            (len(results), 1, min_win), dtype=np.float64
        )
        channel_labels = ["ergodicity"]
        variant = "DE"

    elif isinstance(first, (STResult, CTResult)):
        if isinstance(first, STResult):
            n_ch_first = first.n_channels
            ch_labels = first.channel_labels
            variant = "ST"
        else:
            n_ch_first = first.n_pairs
            ch_labels = first.pair_labels
            variant = "CT"

        n_coeff_first = first.n_coeffs

        for i, r in enumerate(results):
            n_ch = r.n_channels if isinstance(r, STResult) else r.n_pairs
            if n_ch != n_ch_first:
                raise ValueError(
                    f"Channel count mismatch: expected {n_ch_first}, "
                    f"got {n_ch} at index {i}"
                )
            if r.n_coeffs != n_coeff_first:
                raise ValueError(
                    f"Coefficient count mismatch: expected {n_coeff_first}, "
                    f"got {r.n_coeffs} at index {i}"
                )

        min_win = min(r.n_windows for r in results)
        if any(r.n_windows != min_win for r in results):
            warnings.warn(
                f"Window counts vary across subjects. "
                f"Truncating to minimum: {min_win}",
                stacklevel=2,
            )

        coefficients = np.stack(
            [r.coefficients[:, :min_win, :] for r in results], axis=0
        )
        errors = np.stack(
            [r.errors[:, :min_win] for r in results], axis=0
        )
        channel_labels = ch_labels
    else:
        raise ValueError(f"Unsupported result type: {result_type.__name__}")

    return GroupResult(
        coefficients=coefficients,
        errors=errors,
        subject_labels=labels,
        channel_labels=channel_labels,
        params=first.params,
        variant=variant,
    )
