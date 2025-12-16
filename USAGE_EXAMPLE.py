"""
Example usage of the auto-generated DDA Python API

This demonstrates how to use the generated code to run DDA analysis.
"""

from dda_py import (
    BINARY_NAME,
    DDARequest,
    DDARunner,
    Flags,
    generate_select_mask,
    get_variant_by_abbrev,
    parse_select_mask,
)

# Initialize the runner with path to DDA binary
runner = DDARunner(binary_path="run_DDA_AsciiEdf")

# Create an analysis request
request = DDARequest(
    file_path="patient1_S05__01_03.edf",
    channels=[1, 2, 3],  # 0-based channel indices
    variants=["ST", "SY"],  # Which DDA variants to run
    window_length=2048,
    window_step=1024,
    delays=[7, 10],  # Direct delay values for -TAU flag
    model_params=[
        1,
        2,
        10,
    ],  # DDA model encoding for -MODEL flag (default, can be any length)
    # Optional parameters:
    ct_window_length=2,  # Required for CT/CD/DE variants
    ct_window_step=2,
    polynomial_order=4,
    num_tau=2,
)

# Execute DDA analysis
results = runner.run(request)

# Access results for each variant
for variant_name, variant_results in results.items():
    print(f"\n{variant_name} Results:")
    print(
        f"  Matrix shape: {variant_results['num_channels']} × {variant_results['num_timepoints']}"
    )
    print(f"  Stride: {variant_results['stride']}")

    # Access the actual data (structured format)
    channels = variant_results["channels"]  # List of channel data
    print(
        f"  Data: {len(channels)} channels with {len(channels[0]['timepoints'])} timepoints each"
    )
    print(variant_results)


# Example: Using variant metadata
# Get information about a variant
st_variant = get_variant_by_abbrev("ST")
print("\nST Variant Info:")
print(f"  Name: {st_variant.name}")
print(f"  Documentation: {st_variant.documentation}")
print(f"  Stride: {st_variant.stride}")
print(f"  Output suffix: {st_variant.output_suffix}")


# Example: Generating SELECT masks
mask = generate_select_mask(["ST", "CT", "SY"])
print(f"\nSELECT mask for ST+CT+SY: {mask}")

enabled = parse_select_mask(mask)
print(f"Enabled variants: {enabled}")


# Example: Access to CLI constants
print(f"\nBinary name: {BINARY_NAME}")
print(f"Window length flag: {Flags.WINDOW_LENGTH}")
