# dda-py

Python bindings for DDA (Delay Differential Analysis).

The [DDA binary](https://snl.salk.edu/~sfdraeger/dda/) is required. Please download the most recent version from the file server.

## Installation

```bash
pip install dda-py
```

Optional dependencies:

```bash
pip install dda-py[mne]     # MNE-Python integration
pip install dda-py[pandas]  # DataFrame export
pip install dda-py[all]     # All optional deps
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
