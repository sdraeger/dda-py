from pathlib import Path
import tempfile


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
    channel_list: list,
    bounds: tuple[int, int],
    cpu_time: bool,
) -> list[str]:
    """
    Make the command to run DDA.

    Args:
        dda_binary_path: Path to the DDA binary.
        edf_file_name: Path to the EDF file.
        out_file_name: Path to the output file.
        channel_list: List of channels to analyze.
        bounds: Tuple of (start, end) bounds for the analysis.
        cpu_time: Whether to include CPU time in the output.

    Returns:
        list: The command to run DDA.
    """
    command = [
        dda_binary_path,
        "-DATA_FN",
        edf_file_name,
        "-OUT_FN",
        out_file_name,
        "-EDF",
        "-CH_list",
        *channel_list,
        "-dm",
        "4",
        "-order",
        "4",
        "-nr_tau",
        "2",
        "-WL",
        "125",
        "-WS",
        "62",
        "-SELECT",
        "1",
        "0",
        "0",
        "0",
        "-MODEL",
        "1",
        "2",
        "10",
        "-TAU",
        "7",
        "10",
    ]

    if "-1" not in bounds:
        start, end = bounds
        command += ["-StartEnd", str(start), str(end)]

    if cpu_time:
        command.append("-CPUtime")

    return command
