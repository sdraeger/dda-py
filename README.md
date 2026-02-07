# dda-py

Python bindings for DDA (Delay Differential Analysis).

The [DDA binary](https://snl.salk.edu/~sfdraeger/dda/) is required. Please download the most recent version from the file server.

## Installation

```bash
pip install dda-py
```

Optional dependencies:

```bash
pip install 'dda-py[mne]'        # MNE-Python integration
pip install 'dda-py[pandas]'     # DataFrame export
pip install 'dda-py[matplotlib]' # Plotting
pip install 'dda-py[scipy]'      # Window comparison statistics
pip install 'dda-py[mne-bids]'   # BIDS dataset integration
pip install 'dda-py[all]'        # All optional deps
```

## Quick Start (High-Level API)

```python
import numpy as np
from dda_py import run_st

# Analyze a numpy array (n_channels x n_samples)
data = np.random.randn(3, 10000)
result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)

print(result.coefficients.shape)   # (3, n_windows, 3)
print(result.n_channels)           # 3
print(result.n_windows)            # depends on data length
print(result.to_dataframe().head())
```

### MNE-Python Integration

```python
import mne
from dda_py import run_st

raw = mne.io.read_raw_edf("data.edf", preload=True)
result = run_st(raw, delays=(7, 10), wl=200, ws=100)
# sfreq is extracted automatically from the MNE Raw object
```

### Cross-Timeseries Analysis

```python
from dda_py import run_ct

data = np.random.randn(4, 10000)  # 4 channels
result = run_ct(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
print(result.n_pairs)              # 6 (all unique pairs)
print(result.pair_labels)          # ['ch0-ch1', 'ch0-ch2', ...]
```

### Dynamical Ergodicity

```python
from dda_py import run_de

data = np.random.randn(2, 10000)
result = run_de(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
print(result.ergodicity.shape)     # (n_windows,)
```

## Plotting

Requires `pip install 'dda-py[matplotlib]'`.

```python
from dda_py import run_st, plot_coefficients, plot_heatmap, plot_errors, plot_model

result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)

# Coefficient time series per channel
fig = plot_coefficients(result, use_time=True, sfreq=256.0)

# Heatmap (channels x windows) for a single coefficient
fig = plot_heatmap(result, coeff_index=0, cmap="RdBu_r")

# Reconstruction errors over time
fig = plot_errors(result)

# Visualize the model space grid with selected terms highlighted
fig = plot_model([1, 2, 10], num_delays=2, polynomial_order=4)
```

All plotting functions accept an optional `ax` parameter to draw into an existing matplotlib axes, and return a `matplotlib.figure.Figure`.

```python
from dda_py import run_de, plot_ergodicity

result = run_de(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
fig = plot_ergodicity(result, use_time=True, sfreq=256.0)
```

## Batch Processing

Process multiple files in one call:

```python
from dda_py import run_batch, collect_results

# Run DDA on a list of files
results = run_batch(
    ["subj01.edf", "subj02.edf", "subj03.edf"],
    variant="st",
    sfreq=256.0,
    delays=(7, 10),
    wl=200,
    ws=100,
    progress=True,  # shows progress bar (uses tqdm if installed)
)

# Stack results into a single GroupResult for group analysis
group = collect_results(results, labels=["subj01", "subj02", "subj03"])
print(group.coefficients.shape)    # (3, n_channels, n_windows, n_coeffs)
print(group.mean_over_windows())   # (3, n_channels, n_coeffs)
print(group.to_dataframe().head())
```

## Statistics

Group-level statistical analysis between two groups of DDA results.

### Permutation Test

```python
from dda_py import permutation_test

result = permutation_test(
    group_a=results_patients,
    group_b=results_controls,
    n_permutations=10000,
    seed=42,
)

print(result.p_value)              # (n_channels, n_coeffs)
print(result.observed_stat)        # (n_channels, n_coeffs)
print(result.to_dataframe())
```

### Effect Size

```python
from dda_py import compute_effect_size

effect = compute_effect_size(results_patients, results_controls)
print(effect.cohens_d)             # (n_channels, n_coeffs)
print(effect.to_dataframe())
```

### Window Comparison

Requires `pip install 'dda-py[scipy]'`. Compare baseline vs test windows within a single recording:

```python
from dda_py import compare_windows

comp = compare_windows(
    result,
    baseline_windows=slice(0, 10),
    test_windows=slice(10, 20),
    method="ttest",  # or "ranksum"
)
print(comp.p_value)
print(comp.baseline_mean)
print(comp.test_mean)
```

## BIDS Integration

Requires `pip install 'dda-py[mne-bids]'`. Discover and analyze recordings from a BIDS dataset:

```python
from dda_py import find_recordings, run_bids

# List available recordings
recordings = find_recordings("/path/to/bids", datatype="eeg", task="rest")
for rec in recordings:
    print(rec.label)  # e.g. "sub-01_ses-01_task-rest_run-01"

# Run DDA on all matching recordings
results = run_bids(
    "/path/to/bids",
    variant="st",
    datatype="eeg",
    task="rest",
    delays=(7, 10),
    wl=200,
    ws=100,
)
# returns {"sub-01_task-rest": STResult, "sub-02_task-rest": STResult, ...}
```

## Low-Level API

For full control over the DDA binary:

```python
from dda_py import DDARequest, DDARunner

runner = DDARunner()  # auto-discovers binary
request = DDARequest(
    file_path="data.edf",
    channels=[0, 1, 2],
    variants=["ST"],
    window_length=200,
    window_step=100,
    delays=[7, 10],
)
results = runner.run(request)
```

## Model Encoding

Visualize what DDA model indices mean:

```python
from dda_py import visualize_model_space, decode_model_encoding

# Show all monomials for 2 delays, polynomial order 4
print(visualize_model_space(2, 4, highlight_encoding=[1, 2, 10]))

# Decode model [1, 2, 10] to equation
print(decode_model_encoding([1, 2, 10], num_delays=2, polynomial_order=4, format="text"))
# dx/dt = a_1 x_1 + a_2 x_2 + a_3 x_1^4
```

## CLI

```bash
dda --file data.edf --channels 0 1 2 --variants ST --wl 200 --ws 100
dda --file data.edf --channels 0 1 2 --variants ST CT --delays 7 10 -o results.json
```

## Variants

- **ST** - Single Timeseries
- **CT** - Cross Timeseries
- **CD** - Cross Dynamical
- **DE** - Dynamical Ergodicity
- **SY** - Synchrony

## License

MIT
