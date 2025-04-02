import subprocess
from pathlib import Path

import numpy as np

from .utils import create_tempfile, make_dda_command

__all__ = ["run_dda", "init", "DDA_BINARY_PATH"]

DDA_BINARY_PATH = None


def init(dda_binary_path: str):
    """
    Initialize the DDA binary path.

    Args:
        dda_binary_path: Path to the DDA binary.
    """

    if not Path(dda_binary_path).exists():
        raise FileNotFoundError(f"DDA binary not found at {dda_binary_path}")

    global DDA_BINARY_PATH
    DDA_BINARY_PATH = dda_binary_path

    return DDA_BINARY_PATH


def run_dda(
    input_file: str,
    output_file: str | None,
    channel_list: list,
    bounds: tuple[int, int] | None = None,
    cpu_time: bool = False,
):
    """
    Run DDA on the input file and save the output to the output file.
    If the output file is not provided, a temporary file will be created.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file. If None, a temporary file will be created.
        channel_list: List of channels to analyze.
        bounds: Tuple of (start, end) bounds for the analysis.
        cpu_time: Whether to include CPU time in the output.

    Returns:
        numpy.ndarray: The output matrix.
    """

    if output_file is None:
        tempf = create_tempfile(suffix=".dda")
        output_file = tempf.name

    command = make_dda_command(
        DDA_BINARY_PATH,
        input_file,
        output_file,
        channel_list,
        bounds,
        cpu_time,
    )

    # Execute command blocking
    subprocess.run(command)

    ST_filename = f"{output_file}_ST"
    file_path = Path(ST_filename)

    # Read all lines, skip the last one, and write back
    lines = file_path.read_text().splitlines()[:-1]
    file_path.write_text("\n".join(lines))

    Q = np.loadtxt(ST_filename)

    return Q, file_path
