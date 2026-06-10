from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dda_py import (  # noqa: E402
    CD,
    CT,
    DE,
    VARIANT_ORDER,
    Defaults,
    active_variants,
    generate_select_mask,
)


def test_defaults_match_python_binary_api_contract():
    assert Defaults.WINDOW_LENGTH is None
    assert Defaults.WINDOW_STEP is None
    assert Defaults.MODEL_DIMENSION == 3
    assert Defaults.DERIVATIVE_POINTS == 3
    assert Defaults.POLYNOMIAL_ORDER == 4
    assert Defaults.NUM_TAU == 2
    assert list(Defaults.MODEL_PARAMS) == [1, 2, 10]
    assert list(Defaults.DELAYS) == [7, 10]
    assert Defaults.SAMPLING_RATE is None


def test_variant_order_and_active_variants_are_stable():
    assert list(VARIANT_ORDER) == ["ST", "CT", "CD", "RESERVED", "DE", "SY"]
    assert [variant.abbreviation for variant in active_variants()] == [
        "ST",
        "CT",
        "CD",
        "DE",
        "SY",
    ]


def test_select_mask_cases_are_generated_without_external_contract_file():
    cases = {
        "st_only": (["ST"], [1, 0, 0, 0, 0, 0]),
        "ct_only": (["CT"], [0, 1, 0, 0, 0, 0]),
        "cd_only": (["CD"], [0, 0, 1, 0, 0, 0]),
        "de_only": (["DE"], [0, 0, 0, 0, 1, 0]),
        "sy_only": (["SY"], [0, 0, 0, 0, 0, 1]),
        "st_ct_de": (["ST", "CT", "DE"], [1, 1, 0, 0, 1, 0]),
    }

    for name, (flavors, expected_mask) in cases.items():
        assert generate_select_mask(flavors) == expected_mask, name


def test_ct_family_metadata_requires_explicit_ct_window_flags():
    for variant in (CT, CD, DE):
        assert "-WL_CT" in variant.required_params
        assert "-WS_CT" in variant.required_params
