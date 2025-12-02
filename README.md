# dda-py

Python bindings for DDA (Delay Differential Analysis).

## Installation

```bash
pip install dda-py
```

## Usage

```python
from dda_py import DDARequest, DDARunner, ST, generate_select_mask

# Create a DDA request
request = DDARequest(
    input_file="data.csv",
    output_file="results.csv",
    select_mask=generate_select_mask([ST]),  # Select ST variant
)

# Run DDA
runner = DDARunner()
result = runner.run(request)
```

## Variants

The package provides access to all DDA variants:

- `ST` - Standard
- `CT` - Continuous Time
- `CD` - Continuous Data
- `DE` - Differential Equation
- `SY` - Synchrony

## License

MIT
