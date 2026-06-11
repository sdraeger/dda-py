from types import SimpleNamespace

import numpy as np
import pytest

from dda_py import (
    StructureSelectionResult,
    StructureSelectionTrial,
    structure_selection,
)
from dda_py.generated import structure_selection as generated_structure_selection
from dda_py.structure_selection import _score_result, _structure_selection


def _fake_result(error):
    return SimpleNamespace(
        variant_results=[
            SimpleNamespace(
                variant_id="ST",
                errors=np.array([[float(error)]]),
            )
        ]
    )


def test_structure_selection_selects_lowest_error_candidate():
    calls = []
    scores = {
        ((1, 2, 6), (7, 10)): 0.5,
        ((1, 2, 6), (10, 20)): 0.3,
        ((1, 2, 10), (7, 10)): 0.7,
        ((1, 2, 10), (10, 20)): 0.1,
    }

    def run_once(**kwargs):
        model = tuple(kwargs["model"])
        delays = tuple(kwargs["delays"])
        calls.append(kwargs)
        return _fake_result(scores[(model, delays)])

    result = _structure_selection(
        run_once,
        file_path="data.ascii",
        channels=[1, 2],
        binary_path="/tmp/run_DDA_AsciiEdf",
        candidate_models=[[1, 2, 6], [1, 2, 10]],
        candidate_delays=[[7, 10], [10, 20]],
        derivative_points=4,
        order=3,
        WL=3000,
        WS=200,
        input_format="ascii",
    )

    assert isinstance(result, StructureSelectionResult)
    assert result.best_model == [1, 2, 10]
    assert result.best_delays == [10, 20]
    assert result.best_score == 0.1
    assert len(result.trials) == 4
    assert isinstance(result.trials[0], StructureSelectionTrial)
    assert calls[0]["flavors"] == ["ST"]
    assert calls[0]["model"] == [1, 2, 6]
    assert calls[0]["delays"] == [7, 10]
    assert calls[0]["nr_tau"] == 2
    assert calls[0]["WL"] == 3000
    assert calls[0]["WS"] == 200
    assert calls[0]["input_format"] == "ascii"


def test_structure_selection_supports_matrix_models_and_output_directory(tmp_path):
    seen = []
    model = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 1]])

    def run_once(**kwargs):
        seen.append(kwargs)
        return _fake_result(len(seen))

    result = _structure_selection(
        run_once,
        file_path="data.ascii",
        channels=[1],
        candidate_models=[model],
        candidate_delays=[[32, 9]],
        derivative_points=4,
        order=3,
        out_dir=tmp_path,
    )

    assert result.best_model is model
    assert seen[0]["model"] is model
    assert seen[0]["out_fn"] == str(tmp_path / "structure_selection_m1_d1")
    assert result.trials[0].out_fn == seen[0]["out_fn"]


def test_score_result_metrics_and_errors():
    result = SimpleNamespace(
        variant_results=[
            SimpleNamespace(
                variant_id="ST",
                errors=np.array([[1.0, 2.0, 9.0]]),
            )
        ]
    )

    assert _score_result(result, "mean_error") == 4.0
    assert _score_result(result, "median_error") == 2.0
    assert _score_result(result, "minimum_error") == 1.0
    with pytest.raises(ValueError, match="ST"):
        _score_result(
            SimpleNamespace(
                variant_results=[
                    SimpleNamespace(variant_id="CT", errors=np.array([[1.0]]))
                ]
            ),
            "mean_error",
        )
    with pytest.raises(ValueError, match="Unsupported"):
        _score_result(result, "unknown")


def test_structure_selection_is_exported():
    assert callable(structure_selection)
    assert callable(generated_structure_selection)
