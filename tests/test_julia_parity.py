from pathlib import Path

import numpy as np
import pytest

from dda_py import DDARequest, DDARunner, build_command_string, run_DDA
from dda_py.model_encoding import model_matrix_to_encoding


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def test_default_command_matches_julia_shape(tmp_path):
    binary = tmp_path / "run_DDA_AsciiEdf"
    _touch(binary)
    runner = DDARunner(str(binary))
    request = DDARequest("test.edf", [1], flavors=["ST"])

    command = build_command_string(runner, request, "/tmp/default_out")

    assert command == (
        f"sh {binary} -EDF -DATA_FN test.edf -OUT_FN /tmp/default_out "
        "-CH_list 1 -SELECT 1 0 0 0 0 0 -MODEL 1 2 10 -TAU 7 10 "
        "-dm 3 -order 4 -nr_tau 2"
    )
    assert " -WL " not in command
    assert " -WS " not in command
    assert " -SR " not in command


def test_model_matrix_select_format_sampling_and_passthrough_command(tmp_path):
    binary = tmp_path / "run_DDA_AsciiEdf"
    _touch(binary)
    runner = DDARunner(str(binary))
    request = DDARequest(
        "test.custom",
        [1, 2, 1, 4],
        flavors=["ST"],
        select=[1, 1, 0, 0, 0, 0],
        input_format="ascii",
        model=np.array([[0, 0, 1], [0, 0, 2], [1, 1, 1]]),
        derivative_points=4,
        order=3,
        WL=3000,
        WS=200,
        WL_CT=2,
        WS_CT=2,
        time_range=(0, 4000),
        delays=[32, 9],
        sampling_rate=(500, 1000),
        out_fn="test.DDA",
        tau_file="/tmp/tau_values.txt",
        tau2=[11, 12],
        model2=[2, 5, 9],
        no_norm=True,
        WN_list=[4, 8, 12],
    )

    command = build_command_string(runner, request, "test.DDA")

    assert request.variants == ["ST", "CT"]
    assert request.model_terms == [1, 2, 6]
    assert command == (
        f"sh {binary} -ASCII -DATA_FN test.custom -OUT_FN test.DDA "
        "-CH_list 1 2 1 4 -SELECT 1 1 0 0 0 0 -MODEL 1 2 6 "
        "-TAU 32 9 -WL 3000 -WS 200 -dm 4 -order 3 -nr_tau 2 "
        "-WL_CT 2 -WS_CT 2 -StartEnd 0 4000 -SR 500 1000 "
        "-TAU_file /tmp/tau_values.txt -TAU2 11 12 -MODEL2 2 5 9 "
        "-NoNorm -WN_list 4 8 12"
    )


def test_sampling_rate_scalar_tuple_and_validation(tmp_path):
    binary = tmp_path / "run_DDA_AsciiEdf"
    _touch(binary)
    runner = DDARunner(str(binary))

    scalar = DDARequest("test.edf", [1], flavors=["ST"], sampling_rate=500)
    tuple_rate = DDARequest("test.edf", [1], flavors=["ST"], sampling_rate=(500, 500))
    default = DDARequest("test.edf", [1], flavors=["ST"])

    assert " -SR 500" in build_command_string(runner, scalar, "/tmp/scalar")
    assert " -SR 500 500" in build_command_string(runner, tuple_rate, "/tmp/tuple")
    assert " -SR " not in build_command_string(runner, default, "/tmp/default")
    with pytest.raises(ValueError, match="integer-valued"):
        DDARequest("test.edf", [1], flavors=["ST"], sampling_rate=500.5)


def test_run_DDA_repairs_empty_info_and_parses_mixed_st_ct(tmp_path):
    binary = tmp_path / "fake_dda.sh"
    input_file = tmp_path / "input.ascii"
    output_base = tmp_path / "mixed"
    input_file.write_text("A\tB\tC\n1\t2\t3\n4\t5\t6\n", encoding="utf-8")
    binary.write_text(
        """#!/usr/bin/env sh
out=''
st=0
ct=0
channels=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    -OUT_FN) out="$2"; shift 2 ;;
    -SELECT) st="$2"; ct="$3"; shift 7 ;;
    -CH_list)
      shift
      channels=''
      while [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; do
        channels="$channels $1"
        shift
      done
      ;;
    *) shift ;;
  esac
done
: > "$out.info"
if [ "$st" = "1" ]; then
  line='0 10'
  for _ch in $channels; do line="$line 1.0 2.0 3.0 0.1"; done
  printf '%s\\n' "$line" > "${out}_ST"
fi
if [ "$ct" = "1" ]; then
  printf '%s\\n' '0 10 4.0 5.0 6.0 0.2 7.0 8.0 9.0 0.3' > "${out}_CT"
fi
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)

    result = run_DDA(
        file_path=str(input_file),
        channels=[1, 2, 1, 4],
        flavors=["ST", "CT"],
        binary_path=str(binary),
        WL=128,
        WS=100,
        WL_CT=2,
        WS_CT=2,
        out_fn=str(output_base),
    )

    assert [variant.variant_id for variant in result.variant_results] == ["ST", "CT"]
    np.testing.assert_allclose(result.ST, result.variant_results[0].A)
    np.testing.assert_allclose(
        result.CT,
        np.array([[4.0, 5.0, 6.0, 0.2, 7.0, 8.0, 9.0, 0.3]]),
    )
    assert result.channels == ["A", "B", "A", "Channel 4"]
    assert result.variant_results[1].channel_labels == ["A-B", "A-A"]
    assert result.T.tolist() == [[0, 10]]
    assert result.t.tolist() == [14.0]

    info_text = Path(f"{output_base}.info").read_text(encoding="utf-8")
    assert info_text.startswith(f"{binary} -ASCII -DATA_FN")
    assert "-CH_list 1 2 1 4 -SELECT 1 1 0 0 0 0" in info_text
    assert "-WL_CT 2 -WS_CT 2" in info_text


def test_model_matrix_to_encoding_matches_julia_example():
    assert model_matrix_to_encoding(
        [[0, 0, 1], [0, 0, 2], [1, 1, 1]],
        num_delays=2,
        polynomial_order=3,
    ) == [1, 2, 6]
