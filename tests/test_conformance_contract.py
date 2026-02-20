import json
from pathlib import Path

import pytest

# Ensure local package import in editable source layout.
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dda_py import (  # noqa: E402
    CT,
    CD,
    DE,
    VARIANT_ORDER,
    Defaults,
    active_variants,
    generate_select_mask,
)


@pytest.fixture(scope="module")
def contract() -> dict:
    path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "conformance"
        / "dda_conformance_contract.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_defaults_match_contract(contract: dict):
    defaults = contract["defaults"]
    assert Defaults.WINDOW_LENGTH == defaults["window_length"]
    assert Defaults.WINDOW_STEP == defaults["window_step"]
    assert Defaults.MODEL_DIMENSION == defaults["model_dimension"]
    assert Defaults.POLYNOMIAL_ORDER == defaults["polynomial_order"]
    assert Defaults.NUM_TAU == defaults["num_tau"]
    assert list(Defaults.MODEL_PARAMS) == defaults["model_terms"]
    assert list(Defaults.DELAYS) == defaults["delays"]


def test_variant_order_matches_contract(contract: dict):
    assert list(VARIANT_ORDER) == contract["variant_order"]
    assert [v.abbreviation for v in active_variants()] == contract["active_variants"]


def test_select_mask_cases_match_contract(contract: dict):
    for case in contract["select_mask_cases"]:
        got = generate_select_mask(case["variants"])
        assert got == case["mask"], f"select mask mismatch for case {case['name']}"


def test_ct_variants_require_ct_flags(contract: dict):
    by_abbrev = {"CT": CT, "CD": CD, "DE": DE}
    for abbrev in contract["ct_window_required_for"]:
        variant = by_abbrev[abbrev]
        assert "-WL_CT" in variant.required_params
        assert "-WS_CT" in variant.required_params
