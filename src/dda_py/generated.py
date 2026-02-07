"""Backwards compatibility module.

Re-exports everything from the main package for tests that import from dda_py.generated.
"""
from .variants import (
    SPEC_VERSION,
    SELECT_MASK_SIZE,
    BINARY_NAME,
    REQUIRES_SHELL_WRAPPER,
    SHELL_COMMAND,
    SUPPORTED_PLATFORMS,
    ChannelFormat,
    FileType,
    OutputColumns,
    VariantMetadata,
    ST,
    CT,
    CD,
    RESERVED,
    DE,
    SY,
    VARIANT_REGISTRY,
    VARIANT_ORDER,
    SelectMaskPositions,
    get_variant_by_abbrev,
    get_variant_by_suffix,
    get_variant_by_position,
    active_variants,
    generate_select_mask,
    parse_select_mask,
    format_select_mask,
    DEFAULT_DELAYS,
)

from .runner import (
    DDARunner,
    DDARequest,
    Flags,
    Defaults,
)

from .results import (
    STResult,
    CTResult,
    DEResult,
)

from .api import (
    run_st,
    run_ct,
    run_de,
)

from .model_encoding import (
    generate_monomials,
    monomial_to_text,
    monomial_to_latex,
    decode_model_encoding,
    visualize_model_space,
    model_encoding_to_dict,
)

__all__ = [
    "SPEC_VERSION",
    "SELECT_MASK_SIZE",
    "BINARY_NAME",
    "REQUIRES_SHELL_WRAPPER",
    "SHELL_COMMAND",
    "SUPPORTED_PLATFORMS",
    "ChannelFormat",
    "FileType",
    "OutputColumns",
    "VariantMetadata",
    "ST",
    "CT",
    "CD",
    "RESERVED",
    "DE",
    "SY",
    "VARIANT_REGISTRY",
    "VARIANT_ORDER",
    "SelectMaskPositions",
    "get_variant_by_abbrev",
    "get_variant_by_suffix",
    "get_variant_by_position",
    "active_variants",
    "generate_select_mask",
    "parse_select_mask",
    "format_select_mask",
    "DEFAULT_DELAYS",
    "DDARunner",
    "DDARequest",
    "Flags",
    "Defaults",
    "STResult",
    "CTResult",
    "DEResult",
    "run_st",
    "run_ct",
    "run_de",
    "generate_monomials",
    "monomial_to_text",
    "monomial_to_latex",
    "decode_model_encoding",
    "visualize_model_space",
    "model_encoding_to_dict",
]
