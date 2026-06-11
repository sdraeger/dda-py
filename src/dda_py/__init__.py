"""Python interface for Delay Differential Analysis."""

from .api import (
    run_ct,
    run_de,
    run_st,
)
from .batch import (
    GroupResult,
    collect_results,
    run_batch,
)
from .bids import (
    BIDSRecording,
    find_recordings,
    run_bids,
)
from .model_encoding import (
    decode_model_encoding,
    generate_monomials,
    model_encoding_to_dict,
    model_matrix_to_encoding,
    monomial_to_latex,
    monomial_to_text,
    visualize_model_space,
)
from .plotting import (
    plot_coefficients,
    plot_ergodicity,
    plot_errors,
    plot_heatmap,
    plot_model,
)
from .results import (
    CTResult,
    DDAResult,
    DelayParameters,
    DEResult,
    STResult,
    VariantResultData,
    WindowParameters,
)
from .runner import (
    DDARequest,
    DDARunner,
    Defaults,
    Flags,
    build_command_string,
    run_DDA,
)
from .stats import (
    EffectSizeResult,
    PermutationResult,
    WindowComparisonResult,
    compare_windows,
    compute_effect_size,
    permutation_test,
)
from .structure_selection import (
    StructureSelectionResult,
    StructureSelectionTrial,
    structure_selection,
)
from .variants import (
    BINARY_NAME,
    CD,
    CT,
    DE,
    DEFAULT_DELAYS,
    REQUIRES_SHELL_WRAPPER,
    RESERVED,
    SELECT_MASK_SIZE,
    SHELL_COMMAND,
    SPEC_VERSION,
    ST,
    SUPPORTED_PLATFORMS,
    SY,
    VARIANT_ORDER,
    VARIANT_REGISTRY,
    ChannelFormat,
    FileType,
    OutputColumns,
    SelectMaskPositions,
    VariantMetadata,
    active_variants,
    format_select_mask,
    generate_select_mask,
    get_variant_by_abbrev,
    get_variant_by_position,
    get_variant_by_suffix,
    parse_select_mask,
)

__all__ = [
    # Spec constants
    "SPEC_VERSION",
    "SELECT_MASK_SIZE",
    "BINARY_NAME",
    "REQUIRES_SHELL_WRAPPER",
    "SHELL_COMMAND",
    "SUPPORTED_PLATFORMS",
    # Types
    "ChannelFormat",
    "FileType",
    "OutputColumns",
    "VariantMetadata",
    # Variant instances
    "ST",
    "CT",
    "CD",
    "RESERVED",
    "DE",
    "SY",
    # Registries
    "VARIANT_REGISTRY",
    "VARIANT_ORDER",
    "SelectMaskPositions",
    # Functions
    "get_variant_by_abbrev",
    "get_variant_by_suffix",
    "get_variant_by_position",
    "active_variants",
    "generate_select_mask",
    "parse_select_mask",
    "format_select_mask",
    # Delays
    "DEFAULT_DELAYS",
    # Low-level execution
    "DDARunner",
    "DDARequest",
    "Flags",
    "Defaults",
    "build_command_string",
    "run_DDA",
    # Result types
    "DDAResult",
    "DelayParameters",
    "STResult",
    "CTResult",
    "DEResult",
    "VariantResultData",
    "WindowParameters",
    # High-level API
    "run_st",
    "run_ct",
    "run_de",
    # Model encoding
    "generate_monomials",
    "model_matrix_to_encoding",
    "monomial_to_text",
    "monomial_to_latex",
    "decode_model_encoding",
    "visualize_model_space",
    "model_encoding_to_dict",
    # Batch processing
    "GroupResult",
    "run_batch",
    "collect_results",
    # Statistics
    "PermutationResult",
    "EffectSizeResult",
    "WindowComparisonResult",
    "permutation_test",
    "compare_windows",
    "compute_effect_size",
    # Structure selection
    "StructureSelectionTrial",
    "StructureSelectionResult",
    "structure_selection",
    # Plotting
    "plot_coefficients",
    "plot_errors",
    "plot_heatmap",
    "plot_ergodicity",
    "plot_model",
    # BIDS integration
    "BIDSRecording",
    "find_recordings",
    "run_bids",
]

__version__ = "0.4.4"
