"""High-level DDA analysis functions.

Provides ``run_st()``, ``run_ct()``, and ``run_de()`` that accept numpy arrays
or MNE Raw objects and return structured result types with numpy arrays.

Example::

    import numpy as np
    from dda_py import run_st

    data = np.random.randn(3, 10000)
    result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
    print(result.coefficients.shape)   # (3, n_windows, 3)
    print(result.to_dataframe().head())

MNE integration (optional)::

    import mne
    from dda_py import run_st

    raw = mne.io.read_raw_edf("data.edf", preload=True)
    result = run_st(raw, delays=(7, 10), wl=200, ws=100)
"""

from __future__ import annotations

import os
import tempfile
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .results import CTResult, DEResult, STResult
from .runner import DDARequest, DDARunner, Defaults
from .variants import DEFAULT_DELAYS

try:
    import mne
except ImportError:  # pragma: no cover - exercised when optional extra is absent.
    mne = None


def _extract_data(
    data: Union[np.ndarray, Any],
    sfreq: float,
    channels: Optional[Union[List[int], List[str]]] = None,
) -> Tuple[np.ndarray, float, List[str]]:
    """Extract numpy array, sfreq, and labels from input.

    Handles both ``np.ndarray`` and ``mne.io.BaseRaw``.

    Returns:
        (data_array, sfreq, channel_labels)
    """
    # Try MNE first
    if mne is not None and isinstance(data, mne.io.BaseRaw):
        if channels is not None:
            if all(isinstance(c, str) for c in channels):
                picks = mne.pick_channels(data.ch_names, include=channels)
            else:
                picks = [int(c) for c in channels]
        else:
            picks = list(range(len(data.ch_names)))

        raw_data, _ = data[picks, :]
        ch_labels = [data.ch_names[i] for i in picks]
        return raw_data, data.info["sfreq"], ch_labels

    # numpy array path
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"data must be np.ndarray or mne.io.BaseRaw, got {type(data).__name__}"
        )

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(f"data must be 1D or 2D array, got {data.ndim}D")

    if channels is not None:
        if all(isinstance(c, str) for c in channels):
            raise ValueError("String channel names require MNE Raw input")
        ch_indices = [int(c) for c in channels]
        data = data[ch_indices]
        ch_labels = [f"ch{i}" for i in ch_indices]
    else:
        ch_labels = [f"ch{i}" for i in range(data.shape[0])]

    return data, sfreq, ch_labels


def _write_temp_ascii(data: np.ndarray) -> str:
    """Write numpy array (n_channels, n_samples) to a temp ASCII file.

    The DDA binary expects rows = timepoints, columns = channels.

    Returns:
        Path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="dda_input_")
    try:
        np.savetxt(path, data.T, fmt="%.10g", delimiter=" ")
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    else:
        os.close(fd)
    return path


def _raw_to_result_st(
    raw_result: Dict[str, Any],
    channel_labels: List[str],
    params: Dict[str, Any],
) -> STResult:
    """Convert DDARunner raw dict output to STResult."""
    coefficients, errors, window_starts, window_ends = _coefficient_arrays(raw_result)

    return STResult(
        coefficients=coefficients,
        errors=errors,
        window_starts=window_starts,
        window_ends=window_ends,
        channel_labels=channel_labels,
        params=params,
    )


def _raw_to_result_ct(
    raw_result: Dict[str, Any],
    pair_labels: List[str],
    params: Dict[str, Any],
) -> CTResult:
    """Convert DDARunner raw dict output to CTResult."""
    coefficients, errors, window_starts, window_ends = _coefficient_arrays(raw_result)

    return CTResult(
        coefficients=coefficients,
        errors=errors,
        window_starts=window_starts,
        window_ends=window_ends,
        pair_labels=pair_labels,
        params=params,
    )


def _raw_to_result_de(
    raw_result: Dict[str, Any],
    params: Dict[str, Any],
) -> DEResult:
    """Convert DDARunner raw dict output to DEResult."""
    channels_data = raw_result["channels"]
    n_windows = len(channels_data[0]["timepoints"]) if channels_data else 0

    if n_windows == 0:
        return DEResult(
            ergodicity=np.empty(0),
            window_starts=np.empty(0, dtype=np.int64),
            window_ends=np.empty(0, dtype=np.int64),
            params=params,
        )

    ergodicity = np.zeros(n_windows)
    window_starts = np.zeros(n_windows, dtype=np.int64)
    window_ends = np.zeros(n_windows, dtype=np.int64)

    for win_idx, tp in enumerate(channels_data[0]["timepoints"]):
        ergodicity[win_idx] = tp["error"]
        window_starts[win_idx] = tp["window_start"]
        window_ends[win_idx] = tp["window_end"]

    return DEResult(
        ergodicity=ergodicity,
        window_starts=window_starts,
        window_ends=window_ends,
        params=params,
    )


def _coefficient_arrays(
    raw_result: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels_data = raw_result["channels"]
    n_entities = len(channels_data)
    n_windows = len(channels_data[0]["timepoints"]) if channels_data else 0
    if n_windows == 0:
        return (
            np.empty((n_entities, 0, 0)),
            np.empty((n_entities, 0)),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    n_coeffs = len(channels_data[0]["timepoints"][0]["coefficients"])
    coefficients = np.zeros((n_entities, n_windows, n_coeffs))
    errors = np.zeros((n_entities, n_windows))
    window_starts = np.zeros(n_windows, dtype=np.int64)
    window_ends = np.zeros(n_windows, dtype=np.int64)

    for entity_idx, entity_data in enumerate(channels_data):
        for win_idx, timepoint in enumerate(entity_data["timepoints"]):
            coefficients[entity_idx, win_idx, :] = timepoint["coefficients"]
            errors[entity_idx, win_idx] = timepoint["error"]
            if entity_idx == 0:
                window_starts[win_idx] = timepoint["window_start"]
                window_ends[win_idx] = timepoint["window_end"]
    return coefficients, errors, window_starts, window_ends


def _make_params_dict(
    sfreq: float,
    delays: Sequence[int],
    model: List[int],
    wl: Optional[int],
    ws: Optional[int],
    model_dimension: int,
    polynomial_order: int,
    num_tau: int,
) -> Dict[str, Any]:
    return {
        "sfreq": sfreq,
        "delays": list(delays),
        "model": list(model),
        "wl": wl,
        "ws": ws,
        "model_dimension": model_dimension,
        "polynomial_order": polynomial_order,
        "num_tau": num_tau,
    }


def _run_raw_variant(
    *,
    flavor: str,
    data: Union[np.ndarray, Any],
    sfreq: float,
    delays: Sequence[int],
    model: Optional[List[int]],
    wl: Optional[int],
    ws: Optional[int],
    channels: Optional[Union[List[int], List[str]]],
    binary_path: Optional[str],
    model_dimension: int,
    polynomial_order: int,
    num_tau: int,
    ct_wl: Optional[int] = None,
    ct_ws: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    model_terms = list(Defaults.MODEL_PARAMS) if model is None else list(model)
    arr, sfreq_actual, ch_labels = _extract_data(data, sfreq, channels)

    temp_path = _write_temp_ascii(arr)
    try:
        request = DDARequest(
            file_path=temp_path,
            channels=list(range(arr.shape[0])),
            variants=[flavor],
            window_length=wl,
            window_step=ws,
            delays=list(delays),
            model_params=model_terms,
            ct_window_length=ct_wl,
            ct_window_step=ct_ws,
            model_dimension=model_dimension,
            polynomial_order=polynomial_order,
            num_tau=num_tau,
        )
        raw_result = DDARunner(binary_path=binary_path).run(request)[flavor]
    finally:
        os.unlink(temp_path)

    params = _make_params_dict(
        sfreq_actual,
        delays,
        model_terms,
        wl,
        ws,
        model_dimension,
        polynomial_order,
        num_tau,
    )
    return raw_result, ch_labels, params


def run_st(
    data: Union[np.ndarray, Any],
    sfreq: float = 1.0,
    delays: Sequence[int] = DEFAULT_DELAYS,
    model: Optional[List[int]] = None,
    wl: int = Defaults.WINDOW_LENGTH,
    ws: int = Defaults.WINDOW_STEP,
    channels: Optional[Union[List[int], List[str]]] = None,
    binary_path: Optional[str] = None,
    model_dimension: int = Defaults.MODEL_DIMENSION,
    polynomial_order: int = Defaults.POLYNOMIAL_ORDER,
    num_tau: int = Defaults.NUM_TAU,
) -> STResult:
    """Run Single Timeseries (ST) DDA analysis.

    Args:
        data: Input data as ``np.ndarray`` (n_channels, n_samples) or
            ``mne.io.BaseRaw``. A 1D array is treated as a single channel.
        sfreq: Sampling frequency in Hz. Ignored if *data* is an MNE Raw
            object (extracted automatically).
        delays: Delay values (tau), default ``(7, 10)``.
        model: Model encoding indices, default ``[1, 2, 10]``.
        wl: Window length in samples.
        ws: Window step in samples.
        channels: Channel indices (int) or names (str) to analyze.
            If ``None``, all channels are used.
        binary_path: Path to the ``run_DDA_AsciiEdf`` binary, or ``None``
            for auto-discovery.
        model_dimension: Embedding dimension (``-dm`` flag).
        polynomial_order: Polynomial order (``-order`` flag).
        num_tau: Number of tau values (``-nr_tau`` flag).

    Returns:
        :class:`STResult` with numpy arrays.
    """
    raw_result, ch_labels, params = _run_raw_variant(
        flavor="ST",
        data=data,
        sfreq=sfreq,
        delays=delays,
        model=model,
        wl=wl,
        ws=ws,
        channels=channels,
        binary_path=binary_path,
        model_dimension=model_dimension,
        polynomial_order=polynomial_order,
        num_tau=num_tau,
    )
    return _raw_to_result_st(raw_result, ch_labels, params)


def run_ct(
    data: Union[np.ndarray, Any],
    sfreq: float = 1.0,
    delays: Sequence[int] = DEFAULT_DELAYS,
    model: Optional[List[int]] = None,
    wl: int = Defaults.WINDOW_LENGTH,
    ws: int = Defaults.WINDOW_STEP,
    ct_wl: Optional[int] = None,
    ct_ws: Optional[int] = None,
    channels: Optional[Union[List[int], List[str]]] = None,
    binary_path: Optional[str] = None,
    model_dimension: int = Defaults.MODEL_DIMENSION,
    polynomial_order: int = Defaults.POLYNOMIAL_ORDER,
    num_tau: int = Defaults.NUM_TAU,
) -> CTResult:
    """Run Cross-Timeseries (CT) DDA analysis.

    Requires at least 2 channels. Analyzes all unique channel pairs.

    Args:
        data: Input data as ``np.ndarray`` (n_channels, n_samples) or
            ``mne.io.BaseRaw``.
        sfreq: Sampling frequency in Hz.
        delays: Delay values (tau), default ``(7, 10)``.
        model: Model encoding indices, default ``[1, 2, 10]``.
        wl: Window length in samples.
        ws: Window step in samples.
        ct_wl: CT-specific window length (defaults to *wl* if not set).
        ct_ws: CT-specific window step (defaults to *ws* if not set).
        channels: Channel indices or names to analyze.
        binary_path: Path to the DDA binary, or ``None`` for auto-discovery.
        model_dimension: Embedding dimension.
        polynomial_order: Polynomial order.
        num_tau: Number of tau values.

    Returns:
        :class:`CTResult` with numpy arrays.

    Raises:
        ValueError: If fewer than 2 channels are provided.
    """
    arr, sfreq_actual, ch_labels = _extract_data(data, sfreq, channels)

    if arr.shape[0] < 2:
        raise ValueError("CT analysis requires at least 2 channels")

    pair_labels = [f"{left}-{right}" for left, right in combinations(ch_labels, 2)]
    raw_result, _, params = _run_raw_variant(
        flavor="CT",
        data=arr,
        sfreq=sfreq_actual,
        delays=delays,
        model=model,
        wl=wl,
        ws=ws,
        channels=None,
        binary_path=binary_path,
        model_dimension=model_dimension,
        polynomial_order=polynomial_order,
        num_tau=num_tau,
        ct_wl=ct_wl or wl,
        ct_ws=ct_ws or ws,
    )
    params["ct_wl"] = ct_wl or wl
    params["ct_ws"] = ct_ws or ws
    return _raw_to_result_ct(raw_result, pair_labels, params)


def run_de(
    data: Union[np.ndarray, Any],
    sfreq: float = 1.0,
    delays: Sequence[int] = DEFAULT_DELAYS,
    model: Optional[List[int]] = None,
    wl: int = Defaults.WINDOW_LENGTH,
    ws: int = Defaults.WINDOW_STEP,
    ct_wl: Optional[int] = None,
    ct_ws: Optional[int] = None,
    channels: Optional[Union[List[int], List[str]]] = None,
    binary_path: Optional[str] = None,
    model_dimension: int = Defaults.MODEL_DIMENSION,
    polynomial_order: int = Defaults.POLYNOMIAL_ORDER,
    num_tau: int = Defaults.NUM_TAU,
) -> DEResult:
    """Run Dynamical Ergodicity (DE) DDA analysis.

    Args:
        data: Input data as ``np.ndarray`` (n_channels, n_samples) or
            ``mne.io.BaseRaw``.
        sfreq: Sampling frequency in Hz.
        delays: Delay values (tau), default ``(7, 10)``.
        model: Model encoding indices, default ``[1, 2, 10]``.
        wl: Window length in samples.
        ws: Window step in samples.
        ct_wl: CT-specific window length (required by DE variant).
        ct_ws: CT-specific window step (required by DE variant).
        channels: Channel indices or names to analyze.
        binary_path: Path to the DDA binary, or ``None`` for auto-discovery.
        model_dimension: Embedding dimension.
        polynomial_order: Polynomial order.
        num_tau: Number of tau values.

    Returns:
        :class:`DEResult` with numpy arrays.
    """
    raw_result, _, params = _run_raw_variant(
        flavor="DE",
        data=data,
        sfreq=sfreq,
        delays=delays,
        model=model,
        wl=wl,
        ws=ws,
        channels=channels,
        binary_path=binary_path,
        model_dimension=model_dimension,
        polynomial_order=polynomial_order,
        num_tau=num_tau,
        ct_wl=ct_wl or wl,
        ct_ws=ct_ws or ws,
    )
    return _raw_to_result_de(raw_result, params)
