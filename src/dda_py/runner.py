"""Minimal DDA binary runner.

The direct API mirrors the Julia bindings: channels are 1-based, ``flavors``
select DDA outputs, and keyword arguments map directly to binary flags. Legacy
``variants``/``window_length``/``window_step`` names are accepted for existing
Python callers.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .model_encoding import model_matrix_to_encoding
from .results import DDAResult, DelayParameters, VariantResultData, WindowParameters
from .variants import (
    BINARY_NAME,
    REQUIRES_SHELL_WRAPPER,
    SELECT_MASK_SIZE,
    VARIANT_REGISTRY,
    FileType,
    generate_select_mask,
    get_variant_by_abbrev,
    parse_select_mask,
)

DEFAULT_MODEL_PARAMS = [1, 2, 10]
DEFAULT_DERIVATIVE_POINTS = 3
DEFAULT_POLYNOMIAL_ORDER = 4
DEFAULT_NUM_TAU = 2
DEFAULT_WINDOW_LENGTH: Optional[int] = None
DEFAULT_WINDOW_STEP: Optional[int] = None
DEFAULT_DELAYS = (7, 10)

SamplingRate = Optional[Union[int, Tuple[int, int]]]


@dataclass(init=False)
class DDARequest:
    """DDA analysis request parameters.

    Parameters are intentionally close to the binary flags. ``channels`` are
    1-based unless a legacy caller includes channel ``0``, in which case all
    channels are shifted by one for the binary command.
    """

    file_path: str
    channels: List[int]
    binary_channels: List[int]
    variants: List[str]
    flavors: List[str]
    WL: Optional[int]
    WS: Optional[int]
    window_length: Optional[int]
    window_step: Optional[int]
    delays: List[int]
    model_terms: List[int]
    model_params: List[int]
    derivative_points: int
    dm: int
    model_dimension: int
    order: int
    polynomial_order: int
    nr_tau: int
    num_tau: int
    time_range: Optional[Tuple[float, float]]
    ct_window_length: Optional[int]
    ct_window_step: Optional[int]
    select: Optional[List[int]]
    input_format: FileType
    sampling_rate: SamplingRate
    TM: int
    tm: int
    out_fn: Optional[str]
    passthrough_args: List[str]

    def __init__(
        self,
        file_path: str,
        channels: Sequence[int],
        variants: Optional[Sequence[str]] = None,
        *,
        flavors: Optional[Sequence[str]] = None,
        window_length: Optional[int] = None,
        window_step: Optional[int] = None,
        WL: Optional[int] = None,
        WS: Optional[int] = None,
        delays: Optional[Sequence[int]] = None,
        model_params: Optional[Sequence[int]] = None,
        model: Optional[Union[Sequence[int], Sequence[Sequence[int]], np.ndarray]] = None,
        model_encoding: Optional[
            Union[Sequence[int], Sequence[Sequence[int]], np.ndarray]
        ] = None,
        derivative_points: Optional[int] = None,
        dm: Optional[int] = None,
        model_dimension: Optional[int] = None,
        polynomial_order: Optional[int] = None,
        order: Optional[int] = None,
        num_tau: Optional[int] = None,
        nr_tau: Optional[int] = None,
        time_range: Optional[Tuple[float, float]] = None,
        ct_window_length: Optional[int] = None,
        ct_window_step: Optional[int] = None,
        WL_CT: Optional[int] = None,
        WS_CT: Optional[int] = None,
        select: Optional[Sequence[int]] = None,
        input_format: Optional[Union[str, FileType]] = None,
        sampling_rate: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
        TM: Optional[int] = None,
        out_fn: Optional[str] = None,
        tau_file: Optional[str] = None,
        tau2: Optional[Sequence[int]] = None,
        model2: Optional[Sequence[int]] = None,
        no_norm: bool = False,
        WN_list: Optional[Sequence[int]] = None,
    ) -> None:
        self.file_path = str(file_path)
        self.channels = [int(ch) for ch in channels]
        self.binary_channels = _normalize_channels(self.channels)

        self.select = _normalize_select(select)
        self.variants = _resolve_variants(variants, flavors, self.select)
        self.flavors = self.variants

        self.WL = _resolve_optional_int_alias("WL", WL, "window_length", window_length)
        self.WS = _resolve_optional_int_alias("WS", WS, "window_step", window_step)
        self.window_length = self.WL
        self.window_step = self.WS

        self.ct_window_length = _resolve_optional_int_alias(
            "WL_CT", WL_CT, "ct_window_length", ct_window_length
        )
        self.ct_window_step = _resolve_optional_int_alias(
            "WS_CT", WS_CT, "ct_window_step", ct_window_step
        )

        self.delays = _as_int_list("delays", delays or DEFAULT_DELAYS)
        self.nr_tau = _resolve_positive_int_alias("nr_tau", nr_tau, "num_tau", num_tau)
        self.num_tau = self.nr_tau

        self.derivative_points = _resolve_derivative_points(
            derivative_points, dm, model_dimension
        )
        self.dm = self.derivative_points
        self.model_dimension = self.derivative_points
        self.order = _resolve_positive_int_alias(
            "order", order, "polynomial_order", polynomial_order
        )
        self.polynomial_order = self.order

        _validate_custom_model_request(
            model=model,
            model_encoding=model_encoding,
            derivative_points=derivative_points,
            dm=dm,
            model_dimension=model_dimension,
            order=order,
            polynomial_order=polynomial_order,
        )
        self.model_terms = _resolve_model_terms(
            model=model,
            model_encoding=model_encoding,
            model_params=model_params,
            nr_tau=self.nr_tau,
            order=self.order,
        )
        self.model_params = self.model_terms

        self.time_range = _normalize_time_range(time_range)
        self.input_format = _resolve_input_format(self.file_path, input_format)
        self.sampling_rate = _normalize_sampling_rate(sampling_rate)
        self.TM = _resolve_tm(self.delays, TM)
        self.tm = self.TM
        self.out_fn = out_fn
        self.passthrough_args = _build_passthrough_args(
            tau_file=tau_file,
            tau2=tau2,
            model2=model2,
            no_norm=no_norm,
            WN_list=WN_list,
        )


class DDARunner:
    """Execute ``run_DDA_AsciiEdf`` and parse binary outputs."""

    def __init__(self, binary_path: Optional[str] = None):
        from .variants import require_binary

        if binary_path is None:
            self.binary_path = Path(require_binary())
        else:
            self.binary_path = Path(binary_path).expanduser()
            if not self.binary_path.exists():
                raise FileNotFoundError(f"DDA binary not found: {binary_path}")

    def run(self, request: DDARequest) -> Dict[str, Any]:
        """Run DDA and return the legacy raw dictionary format."""

        output_base, cleanup_output = self._execute(request)
        try:
            return self._parse_results(request, output_base)
        finally:
            if cleanup_output:
                self._cleanup_outputs(output_base, request.variants)

    def run_structured(self, request: DDARequest) -> DDAResult:
        """Run DDA and return a Julia-parity ``DDAResult``."""

        output_base, cleanup_output = self._execute(request)
        try:
            variant_results = self._parse_variant_results(request, output_base)
            base_labels = _resolve_requested_channel_labels(
                request.file_path, request.binary_channels, fallback_prefix="Channel "
            )
            if variant_results:
                T = variant_results[0].T
                t = variant_results[0].t
                A = variant_results[0].A
            else:
                T = np.empty((0, 2), dtype=np.int64)
                t = np.empty(0, dtype=float)
                A = np.empty((0, 0), dtype=float)

            return DDAResult(
                id=str(uuid.uuid4()),
                file_path=request.file_path,
                channels=base_labels,
                T=T,
                t=t,
                A=A,
                variant_results=variant_results,
                window_params=WindowParameters(
                    request.WL,
                    request.WS,
                    request.ct_window_length,
                    request.ct_window_step,
                ),
                delay_params=DelayParameters(list(request.delays)),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        finally:
            if cleanup_output:
                self._cleanup_outputs(output_base, request.variants)

    def build_command_string(self, request: DDARequest, output_base: str) -> str:
        return " ".join(self._build_command(request, output_base, Path(request.file_path)))

    def _execute(self, request: DDARequest) -> Tuple[str, bool]:
        input_file = Path(request.file_path).expanduser()
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {request.file_path}")

        output_base, cleanup_output = self._resolve_output_base(request)
        cmd = self._build_command(request, output_base, input_file)

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "DDA execution failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {exc.stderr}"
            ) from exc

        self._ensure_logical_command_info(request, output_base, input_file)
        return output_base, cleanup_output

    def _resolve_output_base(self, request: DDARequest) -> Tuple[str, bool]:
        if request.out_fn:
            return request.out_fn, False
        return os.path.join(tempfile.gettempdir(), f"dda_output_{os.getpid()}_{uuid.uuid4().hex}"), True

    def _build_command(
        self, request: DDARequest, output_base: str, input_file: Path
    ) -> List[str]:
        cmd = _command_prefix(self.binary_path)
        cmd.append(request.input_format.flag)
        cmd.extend(["-DATA_FN", str(input_file)])
        cmd.extend(["-OUT_FN", output_base])

        cmd.append("-CH_list")
        cmd.extend(str(ch) for ch in request.binary_channels)

        mask = request.select if request.select is not None else generate_select_mask(request.variants)
        cmd.append("-SELECT")
        cmd.extend(str(bit) for bit in mask)

        cmd.append("-MODEL")
        cmd.extend(str(term) for term in request.model_terms)

        cmd.append("-TAU")
        cmd.extend(str(delay) for delay in request.delays)

        if request.WL is not None:
            cmd.extend(["-WL", str(request.WL)])
        if request.WS is not None:
            cmd.extend(["-WS", str(request.WS)])

        cmd.extend(["-dm", str(request.derivative_points)])
        cmd.extend(["-order", str(request.order)])
        cmd.extend(["-nr_tau", str(request.nr_tau)])

        if request.ct_window_length is not None:
            cmd.extend(["-WL_CT", str(request.ct_window_length)])
        if request.ct_window_step is not None:
            cmd.extend(["-WS_CT", str(request.ct_window_step)])

        if request.time_range is not None:
            start, stop = request.time_range
            cmd.extend(["-StartEnd", str(int(start)), str(int(stop))])

        sampling_rate_args = _sampling_rate_args(request.sampling_rate)
        if sampling_rate_args:
            cmd.append("-SR")
            cmd.extend(sampling_rate_args)

        cmd.extend(request.passthrough_args)
        return cmd

    def _logical_command(
        self, request: DDARequest, output_base: str, input_file: Path
    ) -> List[str]:
        command = self._build_command(request, output_base, input_file)
        if len(command) >= 2 and command[0] == "sh":
            return command[1:]
        return command

    def _ensure_logical_command_info(
        self, request: DDARequest, output_base: str, input_file: Path
    ) -> None:
        info_path = Path(f"{output_base}.info")
        if info_path.exists() and info_path.read_text(encoding="utf-8").strip():
            return
        logical = self._logical_command(request, output_base, input_file)
        info_path.write_text(" ".join(logical) + "\n", encoding="utf-8")

    def _parse_results(self, request: DDARequest, output_base: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for variant_abbrev in request.variants:
            variant = get_variant_by_abbrev(variant_abbrev)
            if variant is None:
                continue

            output_file = _find_output_file(output_base, variant, variant_abbrev)
            if output_file is None:
                raise FileNotFoundError(
                    f"Expected output file not found for {variant_abbrev}: "
                    f"{output_base}{variant.output_suffix}"
                )

            parsed_data = self._parse_output_file_structured(
                Path(output_file), variant.stride
            )
            channels = parsed_data["channels"]
            results[variant_abbrev] = {
                "channels": channels,
                "num_channels": len(channels),
                "num_timepoints": len(channels[0]["timepoints"]) if channels else 0,
                "stride": variant.stride,
            }
        return results

    def _parse_variant_results(
        self, request: DDARequest, output_base: str
    ) -> List[VariantResultData]:
        base_labels = _resolve_requested_channel_labels(
            request.file_path, request.binary_channels, fallback_prefix="Channel "
        )
        results: List[VariantResultData] = []
        for variant_abbrev in request.variants:
            variant = get_variant_by_abbrev(variant_abbrev)
            if variant is None:
                continue
            output_file = _find_output_file(output_base, variant, variant_abbrev)
            if output_file is None:
                continue
            channels = self._parse_output_file_structured(
                Path(output_file), variant.stride
            )["channels"]
            if not channels:
                continue
            labels = _channel_labels_for_variant(variant, base_labels)
            results.append(_pack_variant_result(variant_abbrev, variant, channels, request, labels))
        return results

    def _parse_output_file_structured(
        self, file_path: Path, stride: int
    ) -> Dict[str, Any]:
        raw_data = _read_numeric_rows(file_path)
        if not raw_data:
            return {"channels": []}

        data_columns = len(raw_data[0]) - 2
        if data_columns <= 0 or data_columns % stride != 0:
            raise ValueError(
                f"Invalid data format: {data_columns} data columns is not "
                f"divisible by stride {stride}"
            )

        num_channels = data_columns // stride
        channels = []
        for channel_idx in range(num_channels):
            timepoints = []
            for row in raw_data:
                start_col = 2 + channel_idx * stride
                values = row[start_col : start_col + stride]
                if len(values) >= 2:
                    coefficients = values[:-1]
                    error = values[-1]
                elif len(values) == 1:
                    coefficients = []
                    error = values[0]
                else:
                    coefficients = []
                    error = 0.0
                timepoints.append(
                    {
                        "window_start": int(row[0]),
                        "window_end": int(row[1]),
                        "coefficients": coefficients,
                        "error": error,
                    }
                )
            channels.append({"channel_index": channel_idx, "timepoints": timepoints})
        return {"channels": channels}

    def _parse_output_file(self, file_path: Path, stride: int) -> List[List[float]]:
        matrix = _read_numeric_rows(file_path)
        if not matrix:
            return []

        extracted = []
        for row in matrix:
            payload = row[2:]
            row_values = []
            for col_idx in range(0, len(payload), stride):
                row_values.append(payload[col_idx])
            extracted.append(row_values)

        if not extracted or not extracted[0]:
            return []
        return [list(col) for col in zip(*extracted)]

    def _cleanup_outputs(self, output_base: str, variants: Sequence[str]) -> None:
        for variant_abbrev in variants:
            variant = get_variant_by_abbrev(variant_abbrev)
            if variant is not None:
                Path(f"{output_base}{variant.output_suffix}").unlink(missing_ok=True)
            Path(f"{output_base}_{variant_abbrev}").unlink(missing_ok=True)
        Path(f"{output_base}.info").unlink(missing_ok=True)


def run_DDA(
    *,
    file_path: str,
    channels: Sequence[int],
    flavors: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    binary_path: Optional[str] = None,
    **kwargs: Any,
) -> DDAResult:
    """Run the DDA binary with Julia-parity keyword arguments."""

    request = DDARequest(
        file_path=file_path,
        channels=channels,
        flavors=flavors,
        variants=variants,
        **kwargs,
    )
    return DDARunner(binary_path=binary_path).run_structured(request)


def build_command_string(
    runner: DDARunner, request: DDARequest, output_base: str
) -> str:
    """Render the shell command as a string for diagnostics and tests."""

    return runner.build_command_string(request, output_base)


class Flags:
    DATA_FILE = "-DATA_FN"
    OUTPUT_FILE = "-OUT_FN"
    CHANNEL_LIST = "-CH_list"
    SELECT_MASK = "-SELECT"
    MODEL = "-MODEL"
    DELAY_VALUES = "-TAU"
    MODEL_DIMENSION = "-dm"
    POLYNOMIAL_ORDER = "-order"
    NUM_TAU = "-nr_tau"
    WINDOW_LENGTH = "-WL"
    WINDOW_STEP = "-WS"
    CT_WINDOW_LENGTH = "-WL_CT"
    CT_WINDOW_STEP = "-WS_CT"
    TIME_BOUNDS = "-StartEnd"
    SAMPLING_RATE = "-SR"


class Defaults:
    MODEL_PARAMS = DEFAULT_MODEL_PARAMS
    MODEL_DIMENSION = DEFAULT_DERIVATIVE_POINTS
    DERIVATIVE_POINTS = DEFAULT_DERIVATIVE_POINTS
    POLYNOMIAL_ORDER = DEFAULT_POLYNOMIAL_ORDER
    NUM_TAU = DEFAULT_NUM_TAU
    WINDOW_LENGTH = DEFAULT_WINDOW_LENGTH
    WINDOW_STEP = DEFAULT_WINDOW_STEP
    WL = DEFAULT_WINDOW_LENGTH
    WS = DEFAULT_WINDOW_STEP
    DELAYS = DEFAULT_DELAYS
    SAMPLING_RATE = None


def _command_prefix(binary_path: Path) -> List[str]:
    if REQUIRES_SHELL_WRAPPER and os.name != "nt":
        return ["sh", str(binary_path)]
    return [str(binary_path)]


def _normalize_channels(channels: Sequence[int]) -> List[int]:
    normalized = [int(ch) for ch in channels]
    if not normalized:
        raise ValueError("At least one channel must be provided")
    if any(ch < 0 for ch in normalized):
        raise ValueError("Channels must be positive 1-based indices, or legacy 0-based indices")
    if any(ch == 0 for ch in normalized):
        return [ch + 1 for ch in normalized]
    return normalized


def _normalize_select(select: Optional[Sequence[int]]) -> Optional[List[int]]:
    if select is None:
        return None
    normalized = [int(bit) for bit in select]
    if len(normalized) != SELECT_MASK_SIZE:
        raise ValueError(f"select must have {SELECT_MASK_SIZE} entries")
    if any(bit not in (0, 1) for bit in normalized):
        raise ValueError("select entries must be 0 or 1")
    if not any(normalized):
        raise ValueError("select must enable at least one variant")
    return normalized


def _resolve_variants(
    variants: Optional[Sequence[str]],
    flavors: Optional[Sequence[str]],
    select: Optional[List[int]],
) -> List[str]:
    if select is not None:
        resolved = parse_select_mask(select)
        if not resolved:
            raise ValueError("select must enable at least one non-reserved variant")
        return resolved

    selected = flavors if flavors is not None else variants
    if selected is None:
        selected = ["ST"]
    resolved = [str(flavor).upper() for flavor in selected]
    if not resolved:
        raise ValueError("At least one flavor must be provided")
    unknown = [flavor for flavor in resolved if get_variant_by_abbrev(flavor) is None]
    if unknown:
        raise ValueError(f"Unknown DDA flavor(s): {unknown}")
    return resolved


def _resolve_optional_int_alias(
    preferred_name: str,
    preferred: Optional[int],
    legacy_name: str,
    legacy: Optional[int],
) -> Optional[int]:
    if preferred is not None and legacy is not None and int(preferred) != int(legacy):
        raise ValueError(
            f"Conflicting {preferred_name!r} and {legacy_name!r} values: "
            f"{preferred} and {legacy}"
        )
    value = preferred if preferred is not None else legacy
    if value is None:
        return None
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{preferred_name} must be positive")
    return normalized


def _resolve_positive_int_alias(
    preferred_name: str,
    preferred: Optional[int],
    legacy_name: str,
    legacy: Optional[int],
) -> int:
    value = _resolve_optional_int_alias(preferred_name, preferred, legacy_name, legacy)
    if value is None:
        if preferred_name == "nr_tau":
            return DEFAULT_NUM_TAU
        if preferred_name == "order":
            return DEFAULT_POLYNOMIAL_ORDER
        raise ValueError(f"{preferred_name} must be provided")
    return value


def _resolve_derivative_points(
    derivative_points: Optional[int],
    dm: Optional[int],
    model_dimension: Optional[int],
) -> int:
    values = [
        int(value)
        for value in (derivative_points, dm, model_dimension)
        if value is not None
    ]
    if not values:
        return DEFAULT_DERIVATIVE_POINTS
    reference = values[0]
    if any(value != reference for value in values[1:]):
        raise ValueError("derivative_points, dm, and model_dimension disagree")
    if reference <= 0:
        raise ValueError("derivative_points must be positive")
    return reference


def _validate_custom_model_request(
    *,
    model: Any,
    model_encoding: Any,
    derivative_points: Optional[int],
    dm: Optional[int],
    model_dimension: Optional[int],
    order: Optional[int],
    polynomial_order: Optional[int],
) -> None:
    if model is None and model_encoding is None:
        return
    has_derivative_points = any(
        value is not None for value in (derivative_points, dm, model_dimension)
    )
    has_order = order is not None or polynomial_order is not None
    if not has_derivative_points or not has_order:
        raise ValueError("Passing model requires explicit derivative_points and order")


def _resolve_model_terms(
    *,
    model: Any,
    model_encoding: Any,
    model_params: Optional[Sequence[int]],
    nr_tau: int,
    order: int,
) -> List[int]:
    selected = model_encoding
    if selected is None:
        selected = model
    if selected is None:
        selected = model_params
    if selected is None:
        return list(DEFAULT_MODEL_PARAMS)

    if _is_matrix_like(selected):
        return model_matrix_to_encoding(
            selected,
            num_delays=nr_tau,
            polynomial_order=order,
        )
    return _as_int_list("model", selected)


def _is_matrix_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 2
    if isinstance(value, (str, bytes)):
        return False
    try:
        rows = list(value)
    except TypeError:
        return False
    if not rows:
        return False
    return isinstance(rows[0], Iterable) and not isinstance(rows[0], (str, bytes))


def _as_int_list(name: str, value: Sequence[int]) -> List[int]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list of integers")
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list of integers") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = []
    for item in values:
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name} must contain only integers")
        normalized.append(int(item))
    return normalized


def _normalize_time_range(
    time_range: Optional[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    if time_range is None:
        return None
    if len(time_range) != 2:
        raise ValueError("time_range must contain exactly two values")
    start, stop = float(time_range[0]), float(time_range[1])
    return start, stop


def _resolve_input_format(
    file_path: str, input_format: Optional[Union[str, FileType]]
) -> FileType:
    if isinstance(input_format, FileType):
        return input_format
    if input_format is not None:
        normalized = str(input_format).strip().lstrip(":-").upper()
        if normalized == "ASCII":
            return FileType.ASCII
        if normalized == "EDF":
            return FileType.EDF
        raise ValueError("input_format must be 'ascii' or 'edf'")

    detected = FileType.from_extension(Path(file_path).suffix)
    return detected if detected is not None else FileType.ASCII


def _normalize_sampling_rate(
    sampling_rate: Optional[Union[int, float, Sequence[Union[int, float]]]]
) -> SamplingRate:
    if sampling_rate is None:
        return None
    if isinstance(sampling_rate, (int, float, np.integer, np.floating)):
        return _normalize_sampling_rate_value("Sampling rate", sampling_rate)
    values = list(sampling_rate)
    if len(values) != 2:
        raise ValueError("sampling_rate must be None, a scalar, or exactly two numbers")
    return (
        _normalize_sampling_rate_value("First sampling rate", values[0]),
        _normalize_sampling_rate_value("Second sampling rate", values[1]),
    )


def _normalize_sampling_rate_value(name: str, value: Union[int, float]) -> int:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if not float(value).is_integer():
        raise ValueError(f"{name} must be integer-valued")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _sampling_rate_args(sampling_rate: SamplingRate) -> List[str]:
    if sampling_rate is None:
        return []
    if isinstance(sampling_rate, tuple):
        return [str(sampling_rate[0]), str(sampling_rate[1])]
    return [str(sampling_rate)]


def _resolve_tm(delays: Sequence[int], TM: Optional[int]) -> int:
    if TM is not None:
        tm = int(TM)
        if tm < 0:
            raise ValueError("TM must be non-negative")
        return tm
    return max(delays) if delays else 0


def _build_passthrough_args(
    *,
    tau_file: Optional[str],
    tau2: Optional[Sequence[int]],
    model2: Optional[Sequence[int]],
    no_norm: bool,
    WN_list: Optional[Sequence[int]],
) -> List[str]:
    args: List[str] = []
    if tau_file is not None:
        args.extend(["-TAU_file", str(tau_file)])
    if tau2 is not None:
        args.append("-TAU2")
        args.extend(str(value) for value in _as_int_list("tau2", tau2))
    if model2 is not None:
        args.append("-MODEL2")
        args.extend(str(value) for value in _as_int_list("model2", model2))
    if no_norm:
        args.append("-NoNorm")
    if WN_list is not None:
        args.append("-WN_list")
        args.extend(str(value) for value in _as_int_list("WN_list", WN_list))
    return args


def _read_numeric_rows(file_path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                values = [float(part) for part in stripped.split()]
            except ValueError:
                continue
            rows.append(values)
    return rows


def _pack_variant_result(
    variant_abbrev: str,
    variant: Any,
    channels: List[Dict[str, Any]],
    request: DDARequest,
    labels: List[str],
) -> VariantResultData:
    n_entities = len(channels)
    n_windows = len(channels[0]["timepoints"]) if channels else 0
    n_coefficients = (
        len(channels[0]["timepoints"][0]["coefficients"]) if n_windows else 0
    )
    coefficients = np.empty((n_entities, n_windows, n_coefficients), dtype=float)
    errors = np.empty((n_entities, n_windows), dtype=float)
    T = np.empty((n_windows, 2), dtype=np.int64)

    for entity_idx, channel_data in enumerate(channels):
        for window_idx, timepoint in enumerate(channel_data["timepoints"]):
            coefficients[entity_idx, window_idx, :] = timepoint["coefficients"]
            errors[entity_idx, window_idx] = timepoint["error"]
            if entity_idx == 0:
                T[window_idx, 0] = _integer_output_index(timepoint["window_start"])
                T[window_idx, 1] = _integer_output_index(timepoint["window_end"])

    A = _binary_payload_matrix(coefficients, errors)
    t = _compute_t_axis(T[:, 0] if n_windows else [], request)
    window_starts, window_ends = _result_window_bounds(request, variant, T)

    return VariantResultData(
        variant_id=variant_abbrev,
        variant_name=variant.name,
        A=A,
        coefficients=coefficients,
        errors=errors,
        T=T,
        t=t,
        window_starts=window_starts,
        window_ends=window_ends,
        channel_labels=labels[:n_entities] if labels else None,
    )


def _integer_output_index(value: Union[int, float]) -> int:
    if not float(value).is_integer():
        raise ValueError(f"DDA output T values must be integer-valued, got {value}")
    return int(value)


def _binary_payload_matrix(coefficients: np.ndarray, errors: np.ndarray) -> np.ndarray:
    n_entities, n_windows, n_coefficients = coefficients.shape
    A = np.empty((n_windows, n_entities * (n_coefficients + 1)), dtype=float)
    col = 0
    for entity_idx in range(n_entities):
        for coefficient_idx in range(n_coefficients):
            A[:, col] = coefficients[entity_idx, :, coefficient_idx]
            col += 1
        A[:, col] = errors[entity_idx, :]
        col += 1
    return A


def _compute_t_axis(raw_T: Sequence[int], request: DDARequest) -> np.ndarray:
    denominator = _sampling_rate_scale(request.sampling_rate)
    return np.array(
        [
            (float(value) + 1 + request.derivative_points + request.TM) / denominator
            for value in raw_T
        ],
        dtype=float,
    )


def _sampling_rate_scale(sampling_rate: SamplingRate) -> float:
    if sampling_rate is None:
        return 1.0
    if isinstance(sampling_rate, tuple):
        return float(max(sampling_rate))
    return float(sampling_rate)


def _result_window_bounds(
    request: DDARequest, variant: Any, T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if T.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if getattr(variant, "required_params", None):
        return T[:, 0].copy(), T[:, 1].copy()
    if request.WL is None or request.WS is None:
        return T[:, 0].copy(), T[:, 1].copy()
    start_offset = int(request.time_range[0]) if request.time_range is not None else 0
    starts = np.array(
        [start_offset + idx * request.WS for idx in range(T.shape[0])],
        dtype=np.int64,
    )
    ends = starts + int(request.WL)
    return starts, ends


def _find_output_file(output_base: str, variant: Any, abbrev: str) -> Optional[Path]:
    output_file = Path(f"{output_base}{variant.output_suffix}")
    if output_file.is_file():
        return output_file
    legacy_file = Path(f"{output_base}_{abbrev}")
    if legacy_file.is_file():
        return legacy_file
    return None


def _channel_labels_for_variant(variant: Any, base_labels: List[str]) -> List[str]:
    channel_format = getattr(variant, "channel_format", None)
    format_value = getattr(channel_format, "value", str(channel_format))
    if format_value == "individual":
        return list(base_labels)
    if format_value == "pairs":
        return [
            f"{base_labels[i]}-{base_labels[j]}"
            for i in range(len(base_labels))
            for j in range(i + 1, len(base_labels))
        ]
    if format_value == "directed_pairs":
        return [
            f"{base_labels[i]}->{base_labels[j]}"
            for i in range(len(base_labels))
            for j in range(len(base_labels))
            if i != j
        ]
    return list(base_labels)


def _resolve_requested_channel_labels(
    file_path: str, channels: Sequence[int], *, fallback_prefix: str
) -> List[str]:
    inferred = _infer_input_channel_labels(file_path)
    labels: List[str] = []
    for channel in channels:
        idx = int(channel)
        if inferred is not None and idx <= len(inferred):
            label = _sanitize_channel_label(inferred[idx - 1])
            if label:
                labels.append(label)
                continue
        labels.append(f"{fallback_prefix}{idx}")
    return labels


def _infer_input_channel_labels(file_path: str) -> Optional[List[str]]:
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".edf":
            return _read_edf_channel_labels(path)
        return _read_ascii_channel_labels(path)
    except OSError:
        return None


def _read_edf_channel_labels(path: Path) -> Optional[List[str]]:
    with open(path, "rb") as handle:
        fixed_header = handle.read(256)
        if len(fixed_header) != 256:
            return None
        try:
            signal_count = int(fixed_header[252:256].decode("ascii").strip())
        except ValueError:
            return None
        if signal_count <= 0:
            return None
        labels = []
        for _ in range(signal_count):
            field = handle.read(16)
            if len(field) != 16:
                return None
            labels.append(_sanitize_channel_label(field.decode("latin1")))
        return labels if any(labels) else None


def _read_ascii_channel_labels(path: Path) -> Optional[List[str]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = _split_ascii_fields(stripped)
            if not fields:
                continue
            if all(_is_numeric_field(field) for field in fields):
                return None
            return fields
    return None


def _split_ascii_fields(line: str) -> List[str]:
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return [_sanitize_channel_label(part) for part in parts]


def _is_numeric_field(field: str) -> bool:
    try:
        float(field.strip())
    except ValueError:
        return False
    return bool(field.strip())


def _sanitize_channel_label(label: str) -> str:
    return label.replace("\0", " ").strip().strip("\"'")
