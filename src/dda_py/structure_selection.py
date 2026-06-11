"""Structure selection over explicit DDA model and delay candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from .results import DDAResult
from .runner import run_DDA


@dataclass
class StructureSelectionTrial:
    """One evaluated structure-selection candidate."""

    model: Any
    delays: List[int]
    score: float
    result: DDAResult
    out_fn: Optional[str]


@dataclass
class StructureSelectionResult:
    """Best candidate and all evaluated structure-selection trials."""

    best_model: Any
    best_delays: List[int]
    best_score: float
    best_result: DDAResult
    trials: List[StructureSelectionTrial]


def structure_selection(**kwargs: Any) -> StructureSelectionResult:
    """Evaluate explicit ``-MODEL``/``-TAU`` candidates and select the lowest ST error."""

    return _structure_selection(run_DDA, **kwargs)


def _structure_selection(
    run_once: Callable[..., DDAResult],
    *,
    file_path: str,
    channels: Sequence[int],
    candidate_models: Any,
    candidate_delays: Any,
    binary_path: Optional[str] = None,
    derivative_points: Optional[int] = None,
    order: Optional[int] = None,
    WL: Optional[int] = None,
    WS: Optional[int] = None,
    input_format: Optional[str] = None,
    metric: str = "mean_error",
    out_dir: Optional[Union[Path, str]] = None,
    **kwargs: Any,
) -> StructureSelectionResult:
    if derivative_points is None:
        raise ValueError("derivative_points is required for structure selection")
    if order is None:
        raise ValueError("order is required for structure selection")

    models = _normalize_candidate_models(candidate_models)
    delay_sets = _normalize_candidate_delays(candidate_delays)
    output_root = _output_root(out_dir)
    trials: list[StructureSelectionTrial] = []
    best_trial: Optional[StructureSelectionTrial] = None

    for model_idx, model in enumerate(models, start=1):
        for delay_idx, delays in enumerate(delay_sets, start=1):
            out_fn = _trial_out_fn(output_root, model_idx, delay_idx)
            result = run_once(
                file_path=file_path,
                channels=channels,
                flavors=["ST"],
                binary_path=binary_path,
                model=model,
                delays=delays,
                derivative_points=int(derivative_points),
                order=int(order),
                nr_tau=len(delays),
                WL=WL,
                WS=WS,
                input_format=input_format,
                out_fn=out_fn,
                **kwargs,
            )
            trial = StructureSelectionTrial(
                model=model,
                delays=delays,
                score=_score_result(result, metric),
                result=result,
                out_fn=out_fn,
            )
            trials.append(trial)
            if best_trial is None or trial.score < best_trial.score:
                best_trial = trial

    if best_trial is None:
        raise ValueError("No structure-selection candidates were evaluated")

    return StructureSelectionResult(
        best_model=best_trial.model,
        best_delays=best_trial.delays,
        best_score=best_trial.score,
        best_result=best_trial.result,
        trials=trials,
    )


def _score_result(result: DDAResult, metric: str) -> float:
    errors = np.asarray(_find_st_result(result).errors, dtype=float).ravel()
    if errors.size == 0:
        raise ValueError("No ST error values found")

    if metric == "mean_error":
        return float(np.mean(errors))
    if metric == "median_error":
        return float(np.median(errors))
    if metric == "minimum_error":
        return float(np.min(errors))
    raise ValueError(f"Unsupported structure-selection metric: {metric}")


def _find_st_result(result: DDAResult) -> Any:
    for variant in result.variant_results:
        if variant.variant_id == "ST":
            return variant
    raise ValueError("DDA result does not contain ST output")


def _normalize_candidate_models(candidate_models: Any) -> list[Any]:
    if isinstance(candidate_models, np.ndarray):
        if candidate_models.ndim in {1, 2}:
            return [candidate_models]
        raise ValueError("candidate_models arrays must be one- or two-dimensional")
    if _is_int_sequence(candidate_models):
        return [list(map(int, candidate_models))]

    models = []
    for model in candidate_models:
        if isinstance(model, np.ndarray):
            if model.ndim not in {1, 2}:
                raise ValueError(
                    "candidate model arrays must be one- or two-dimensional"
                )
            models.append(model)
        elif _is_int_sequence(model):
            models.append(list(map(int, model)))
        elif _is_matrix_sequence(model):
            models.append(model)
        else:
            raise ValueError(
                "candidate_models entries must be integer vectors or matrices"
            )
    if not models:
        raise ValueError("candidate_models must contain at least one model")
    return models


def _normalize_candidate_delays(candidate_delays: Any) -> list[list[int]]:
    if _is_int_sequence(candidate_delays):
        return [list(map(int, candidate_delays))]

    delays = []
    for delay_set in candidate_delays:
        if not _is_int_sequence(delay_set):
            raise ValueError("candidate_delays entries must be integer vectors")
        delays.append(list(map(int, delay_set)))
    if not delays:
        raise ValueError("candidate_delays must contain at least one delay set")
    return delays


def _is_int_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        values = list(value)
    except TypeError:
        return False
    return bool(values) and all(isinstance(item, (int, np.integer)) for item in values)


def _is_matrix_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        rows = list(value)
    except TypeError:
        return False
    return bool(rows) and all(_is_int_sequence(row) for row in rows)


def _output_root(out_dir: Optional[Union[Path, str]]) -> Optional[Path]:
    if out_dir is None:
        return None
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _trial_out_fn(
    output_root: Optional[Path],
    model_idx: int,
    delay_idx: int,
) -> Optional[str]:
    if output_root is None:
        return None
    return str(output_root / f"structure_selection_m{model_idx}_d{delay_idx}")
