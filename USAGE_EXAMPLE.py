"""
Example usage of the DDA Python API

This demonstrates both the high-level and low-level APIs.
"""

# === High-Level API (recommended) ===

import numpy as np

from dda_py import (
    BINARY_NAME,
    DDARequest,
    DDARunner,
    Defaults,
    Flags,
    decode_model_encoding,
    generate_select_mask,
    get_variant_by_abbrev,
    parse_select_mask,
    run_ct,
    run_DDA,
    run_de,
    run_st,
    visualize_model_space,
)

# Generate synthetic data: 3 channels, 10000 samples
data = np.random.randn(3, 10000)

# Single-timeseries analysis
st_result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
print(f"ST coefficients shape: {st_result.coefficients.shape}")
print(f"ST windows: {st_result.n_windows}")
print(f"ST channels: {st_result.channel_labels}")

# Export to DataFrame (requires pandas)
df = st_result.to_dataframe()
print(df.head())

# Cross-timeseries analysis (requires >= 2 channels)
ct_result = run_ct(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
print(f"\nCT pairs: {ct_result.pair_labels}")
print(f"CT coefficients shape: {ct_result.coefficients.shape}")

# Dynamical ergodicity
de_result = run_de(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
print(f"\nDE ergodicity shape: {de_result.ergodicity.shape}")


# === Low-Level API ===

runner = DDARunner()  # auto-discovers binary

request = DDARequest(
    file_path="patient1_S05__01_03.edf",
    channels=[1, 2, 3],  # 1-based channel indices
    flavors=["ST", "SY"],  # Which DDA flavors to run
    WL=200,
    WS=100,
    delays=[7, 10],
    model=[1, 2, 10],
    derivative_points=4,
    order=4,
    nr_tau=2,
)

results = runner.run(request)

for variant_name, variant_results in results.items():
    print(f"\n{variant_name} Results:")
    print(
        f"  Channels: {variant_results['num_channels']}, "
        f"Timepoints: {variant_results['num_timepoints']}"
    )
    print(f"  Stride: {variant_results['stride']}")

structured = run_DDA(
    file_path="patient1_S05__01_03.edf",
    channels=[1, 2, 3],
    flavors=["ST", "SY"],
    WL=200,
    WS=100,
)
print(f"\nStructured flavors: {[v.variant_id for v in structured.variant_results]}")

# === Model Encoding ===

# Visualize all available monomials
print(visualize_model_space(2, 4, highlight_encoding=[1, 2, 10]))

# Decode a model encoding to an equation
print(decode_model_encoding([1, 2, 10], 2, 4, format="text"))

# Variant metadata
st_variant = get_variant_by_abbrev("ST")
print(f"\nST Variant: {st_variant.name}")
print(f"  Documentation: {st_variant.documentation}")
print(f"  Stride: {st_variant.stride}")

# SELECT masks
mask = generate_select_mask(["ST", "CT", "SY"])
print(f"\nSELECT mask for ST+CT+SY: {mask}")
print(f"Enabled variants: {parse_select_mask(mask)}")

# Defaults
print(f"\nBinary name: {BINARY_NAME}")
print(f"Default model dimension: {Defaults.MODEL_DIMENSION}")
print(f"Default model params: {Defaults.MODEL_PARAMS}")
print(f"Window length flag: {Flags.WINDOW_LENGTH}")
