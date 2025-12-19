# dda-py

Python bindings for DDA (Delay Differential Analysis).

The [DDA binary](https://snl.salk.edu/~sfdraeger/dda/) is required. Please download the most recent version from the file server.

## Installation

```bash
pip install dda-py
```

## Usage

### Basic Example

```python
from dda_py import DDARequest, DDARunner

# Initialize the runner with path to DDA binary
runner = DDARunner(binary_path="run_DDA_AsciiEdf")

# Create an analysis request
request = DDARequest(
    file_path="patient1_S05__01_03.edf",
    channels=[1, 2, 3],  # 0-based channel indices
    variants=["ST", "SY"],  # Which DDA variants to run
    window_length=2048,
    window_step=1024,
    delays=[7, 10],  # Delay values for -TAU flag
)

# Execute DDA analysis
results = runner.run(request)

# Access results for each variant
for variant_name, variant_results in results.items():
    print(f"\n{variant_name} Results:")
    print(f"  Matrix shape: {variant_results['num_channels']} × {variant_results['num_timepoints']}")
    print(f"  Stride: {variant_results['stride']}")

    # Access the actual data
    channels = variant_results["channels"]
    print(f"  Data: {len(channels)} channels with {len(channels[0]['timepoints'])} timepoints each")
```

### Advanced Options

```python
# Full configuration example with all optional parameters
request = DDARequest(
    file_path="patient1_S05__01_03.edf",
    channels=[1, 2, 3],
    variants=["ST", "CT", "SY"],
    window_length=2048,
    window_step=1024,
    delays=[7, 10],
    model_params=[1, 2, 10],  # DDA model encoding for -MODEL flag
    # Optional parameters:
    ct_window_length=2,  # Required for CT/CD/DE variants
    ct_window_step=2,
    polynomial_order=4,
    num_tau=2,
)
```

### Working with Variant Metadata

```python
from dda_py import get_variant_by_abbrev, generate_select_mask, parse_select_mask

# Get information about a variant
st_variant = get_variant_by_abbrev("ST")
print(f"Name: {st_variant.name}")
print(f"Documentation: {st_variant.documentation}")
print(f"Stride: {st_variant.stride}")

# Generate SELECT masks for multiple variants
mask = generate_select_mask(["ST", "CT", "SY"])
print(f"SELECT mask: {mask}")

# Parse a mask to see which variants are enabled
enabled = parse_select_mask(mask)
print(f"Enabled variants: {enabled}")
```

## Variants

The package provides access to all DDA variants:

- `ST` - Single Timeseries
- `CT` - Cross Timeseries
- `CD` - Cross Dynamical
- `DE` - Dynamical Ergodicity
- `SY` - Synchrony

## License

MIT
