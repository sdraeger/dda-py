# `dda-py`: Python Wrapper for Delay Differential Analysis

A Python wrapper for DDA with native support for APE (Actually Portable Executable) binaries, enabling cross-platform execution without platform-specific binaries.

## Features

- **APE Binary Support**: Native support for Actually Portable Executable binaries
- **Cross-Platform**: Works on Windows, macOS, and Linux with the same APE binary
- **Async Support**: Both synchronous and asynchronous execution
- **Easy Integration**: Simple Python API for DDA analysis

## Installation

Install the package from PyPI:

```bash
pip install dda-py
```

## Usage

### Basic Usage

```python
import dda_py

# Initialize with APE binary path
dda_py.init("./run_DDA_AsciiEdf")

# Run DDA analysis
Q, output_path = dda_py.run_dda(
    input_file="data.edf",
    channel_list=["1", "2", "3"]
)

print(f"Result shape: {Q.shape}")  # channels × time windows
```

### Using DDARunner Class

```python
from dda_py import DDARunner

# Create runner instance
runner = DDARunner("./run_DDA_AsciiEdf")

# Run analysis with options
Q, output_path = runner.run(
    input_file="data.edf",
    channel_list=["1", "2", "3"],
    bounds=(1000, 5000),  # Optional time bounds
    cpu_time=True         # Enable CPU timing
)
```

### Async Usage

```python
import asyncio
from dda_py import DDARunner

async def analyze_data():
    runner = DDARunner("./run_DDA_AsciiEdf")
    Q, output_path = await runner.run_async(
        input_file="data.edf",
        channel_list=["1", "2", "3"]
    )
    return Q

# Run async
result = asyncio.run(analyze_data())
```

## APE Binary Support

This package is designed to work with APE (Actually Portable Executable) binaries. APE binaries:

- Run on Windows, macOS, and Linux without modification
- No need for platform-specific binaries
- Automatic platform detection and execution

The package automatically handles APE binary execution across different platforms using the appropriate shell interpreter when needed.

## Requirements

- Python 3.6+
- NumPy >= 1.19.0
- DDA APE binary (place in your working directory)
