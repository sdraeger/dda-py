from typing import List, Tuple, Optional, Dict, Union
import subprocess
from pathlib import Path
import asyncio
import tempfile

import numpy as np


# Constants for fixed parameters
BASE_PARAMS: Dict[str, Union[str, List[str]]] = {
    "-dm": "4",
    "-order": "4",
    "-nr_tau": "2",
    "-WL": "125",
    "-WS": "62",
    "-SELECT": ["1", "0", "0", "0"],
    "-MODEL": ["1", "2", "10"],
    "-TAU": ["7", "10"],
}

__all__ = ["DDARunner", "init", "DDA_BINARY_PATH"]

DDA_BINARY_PATH: Optional[str] = None


def init(dda_binary_path: str) -> str:
    """Initialize the DDA binary path."""

    if not Path(dda_binary_path).exists():
        raise FileNotFoundError(f"DDA binary not found at {dda_binary_path}")

    global DDA_BINARY_PATH
    DDA_BINARY_PATH = dda_binary_path
    print(f"Set DDA_BINARY_PATH to {DDA_BINARY_PATH}")

    return DDA_BINARY_PATH


class DDARunner:
    """Handles DDA execution, both synchronously and asynchronously."""

    def __init__(self, binary_path: str = DDA_BINARY_PATH):
        if not binary_path:
            raise ValueError(
                "DDA binary path must be initialized via init() or provided."
            )
        self.binary_path = binary_path

    @staticmethod
    def _create_tempfile(subdir: Optional[str] = None, **kwargs) -> Path:
        """Create a temporary file in the .dda directory."""

        d = Path(tempfile.gettempdir()) / ".dda" / (subdir or "")
        d.mkdir(parents=True, exist_ok=True)
        tempf = tempfile.NamedTemporaryFile(dir=d, delete=False, **kwargs)

        return Path(tempf.name)

    @staticmethod
    def _make_command(
        input_file: str,
        output_file: str,
        channel_list: List[str],
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
    ) -> List[str]:
        """Construct a command list for DDA execution."""

        command = [
            DDA_BINARY_PATH,
            "-DATA_FN",
            input_file,
            "-OUT_FN",
            output_file,
            "-EDF",
            "-CH_list",
            *list(map(str, channel_list)),
        ]

        for flag, value in BASE_PARAMS.items():
            command.extend([flag, *value] if isinstance(value, list) else [flag, value])

        if bounds:
            command.extend(["-StartEnd", str(bounds[0]), str(bounds[1])])

        if cpu_time:
            command.append("-CPUtime")

        return command

    @staticmethod
    def _process_output(output_path: Path) -> Tuple[np.ndarray, Path]:
        """Process the DDA output file and load the result."""

        st_path = output_path.with_name(f"{output_path.stem}_ST")
        lines = st_path.read_text().splitlines()[:-1]
        st_path.write_text("\n".join(lines))

        return np.loadtxt(st_path), st_path

    def _prepare_execution(
        self,
        input_file: str,
        output_file: Optional[str],
        channel_list: List[str],
        bounds: Optional[Tuple[int, int]],
        cpu_time: bool,
    ) -> Tuple[List[str], Path]:
        """Prepare command and output path for execution."""

        output_path = Path(output_file) if output_file else self._create_tempfile()
        command = self._make_command(
            input_file, str(output_path), channel_list, bounds, cpu_time
        )

        return command, output_path

    def run(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        channel_list: List[str] = [],
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
        raise_on_error: bool = False,
    ) -> Tuple[np.ndarray, Path]:
        """Run DDA synchronously."""

        command, output_path = self._prepare_execution(
            input_file, output_file, channel_list, bounds, cpu_time
        )
        process = subprocess.run(command)

        if raise_on_error and process.returncode != 0:
            stderr = process.stderr if process.stderr else b""
            raise subprocess.CalledProcessError(
                process.returncode, command, stderr.decode()
            )

        return self._process_output(output_path)

    async def run_async(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        channel_list: List[str] = [],
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
        raise_on_error: bool = False,
    ) -> Tuple[np.ndarray, Path]:
        """Run DDA asynchronously."""

        command, output_path = self._prepare_execution(
            input_file, output_file, channel_list, bounds, cpu_time
        )
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        await process.wait()

        if raise_on_error and process.returncode != 0:
            stderr = await process.stderr.read()
            raise subprocess.CalledProcessError(
                process.returncode, command, stderr.decode()
            )

        return self._process_output(output_path)


# For backward compatibility or simpler usage
def run_dda(*args, **kwargs) -> Tuple[np.ndarray, Path]:
    """Synchronous DDA execution (global instance)."""
    return DDARunner(DDA_BINARY_PATH).run(*args, **kwargs)


async def run_dda_async(*args, **kwargs) -> Tuple[np.ndarray, Path]:
    """Asynchronous DDA execution (global instance)."""
    return await DDARunner(DDA_BINARY_PATH).run_async(*args, **kwargs)
