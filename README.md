# `dda-py`: Python Wrapper for Delay Differential Analysis

## Installation

Install the package from PyPI:

```bash
pip install dda-py
```

Download the appropriate DDA binary from [here](https://github.com/dda-py/dda-py/releases) (select the correct one for your platform) and place it in the root directory of the
project.
Define the path to the binary in the `dda_binary_path` variable in the `init` function.

## Usage

```python
from dda_py import dda

dda.init(dda_binary_path="path/to/dda")

Q = dda.run_dda(
    input_file="path/to/input/file",
    output_file="path/to/output/file",
    channel_list=[1, 2, 3],
)
```
