import tempfile
from pathlib import Path
from typing import List, Tuple

# Constants for fixed parameters
BASE_PARAMS = {
    "-dm": "4",
    "-order": "4",
    "-nr_tau": "2",
    "-WL": "125",
    "-WS": "62",
    "-SELECT": ["1", "0", "0", "0"],
    "-MODEL": ["1", "2", "10"],
    "-TAU": ["7", "10"],
}


def create_tempfile(subdir: str, **kwargs):
    """
    Create a temporary file in the dda directory.

    Args:
        subdir: Subdirectory to create the file in.
        **kwargs: Additional arguments to pass to the tempfile.NamedTemporaryFile constructor.

    Returns:
        tempfile.NamedTemporaryFile: The created temporary file.
    """
    d = Path(tempfile.gettempdir()) / ".dda" / subdir
    d.mkdir(parents=True, exist_ok=True)
    tempf = tempfile.NamedTemporaryFile(dir=d, delete=False, **kwargs)
    return tempf


def make_dda_command(
    dda_binary_path: str,
    edf_file_name: str,
    out_file_name: str,
    channel_list: List[str],
    bounds: Tuple[int, int],
    cpu_time: bool,
) -> List[str]:
    """
    Constructs a command list for DDA binary execution.

    Args:
        dda_binary_path: Path to the DDA binary
        edf_file_name: Input EDF file name
        out_file_name: Output file name
        channel_list: List of channel identifiers
        bounds: Tuple of (start, end) time bounds
        cpu_time: Flag to include CPU time measurement

    Returns:
        List of command arguments
    """
    # Base command components
    command = [
        dda_binary_path,
        "-DATA_FN",
        edf_file_name,
        "-OUT_FN",
        out_file_name,
        "-EDF",
        "-CH_list",
        *channel_list,
    ]

    # Add fixed parameters
    for flag, value in BASE_PARAMS.items():
        if isinstance(value, list):
            command.extend([flag, *value])
        else:
            command.extend([flag, value])

    # Add optional bounds
    if "-1" not in map(str, bounds):  # Convert bounds to strings for comparison
        command.extend(["-StartEnd", str(bounds[0]), str(bounds[1])])

    # Add CPU time flag if requested
    if cpu_time:
        command.append("-CPUtime")

    return command
