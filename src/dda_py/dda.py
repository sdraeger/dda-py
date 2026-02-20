import asyncio
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Default analysis parameters.
DEFAULT_MODEL_DIMENSION = 4
DEFAULT_POLYNOMIAL_ORDER = 4
DEFAULT_NUM_TAU = 2
DEFAULT_WINDOW_LENGTH = 200
DEFAULT_WINDOW_STEP = 100
DEFAULT_SELECT_MASK = [1, 0, 0, 0]  # ST enabled by default
DEFAULT_MODEL_PARAMS = [1, 2, 10]
DEFAULT_DELAYS = [7, 10]

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

    def _get_ape_command(self, binary_path: str) -> List[str]:
        """Get the proper command to execute an APE binary."""
        # On macOS and Unix, APE binaries need to be executed through sh
        system = platform.system()

        if system == "Darwin":  # macOS
            # On macOS, APE binaries need to be executed through sh
            return ["sh", binary_path]
        elif system == "Linux":
            # On Linux, APE binaries can usually run directly, but sh works too
            return [binary_path]
        elif system == "Windows":
            # On Windows, APE binaries run directly
            return [binary_path]
        else:
            # Default to using sh for safety
            return ["sh", binary_path]

    def _make_command(
        self,
        input_file: str,
        output_file: str,
        channel_list: List[int],
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
        select_mask: Optional[List[int]] = None,
        model_params: Optional[List[int]] = None,
        delays: Optional[List[int]] = None,
        window_length: Optional[int] = None,
        window_step: Optional[int] = None,
        ct_window_length: Optional[int] = None,
        ct_window_step: Optional[int] = None,
        model_dimension: Optional[int] = None,
        polynomial_order: Optional[int] = None,
        num_tau: Optional[int] = None,
    ) -> List[str]:
        """Construct a command list for DDA execution."""

        # Get the proper command prefix for APE execution
        command = self._get_ape_command(self.binary_path)

        # Add DDA-specific arguments
        command.extend(
            [
                "-DATA_FN",
                input_file,
                "-OUT_FN",
                output_file,
                "-EDF",
                "-CH_list",
                *[str(ch) for ch in channel_list],
            ]
        )

        effective_model_dimension = (
            model_dimension
            if model_dimension is not None
            else DEFAULT_MODEL_DIMENSION
        )
        effective_polynomial_order = (
            polynomial_order
            if polynomial_order is not None
            else DEFAULT_POLYNOMIAL_ORDER
        )
        effective_num_tau = num_tau if num_tau is not None else DEFAULT_NUM_TAU
        effective_window_length = (
            window_length if window_length is not None else DEFAULT_WINDOW_LENGTH
        )
        effective_window_step = (
            window_step if window_step is not None else DEFAULT_WINDOW_STEP
        )
        effective_select_mask = (
            select_mask if select_mask is not None else DEFAULT_SELECT_MASK
        )
        effective_model_params = (
            model_params if model_params is not None else DEFAULT_MODEL_PARAMS
        )
        effective_delays = delays if delays is not None else DEFAULT_DELAYS

        command.extend(
            [
                "-dm",
                str(effective_model_dimension),
                "-order",
                str(effective_polynomial_order),
                "-nr_tau",
                str(effective_num_tau),
                "-WL",
                str(effective_window_length),
                "-WS",
                str(effective_window_step),
                "-SELECT",
                *[str(v) for v in effective_select_mask],
                "-MODEL",
                *[str(v) for v in effective_model_params],
                "-TAU",
                *[str(v) for v in effective_delays],
            ]
        )

        if ct_window_length is not None:
            command.extend(["-WL_CT", str(ct_window_length)])
        if ct_window_step is not None:
            command.extend(["-WS_CT", str(ct_window_step)])

        if bounds:
            command.extend(["-StartEnd", str(bounds[0]), str(bounds[1])])

        if cpu_time:
            command.append("-CPUtime")

        return command

    @staticmethod
    def _process_output(output_path: Path) -> Tuple[np.ndarray, Path]:
        """Process the DDA output file and load the result."""

        # Handle the case where DDA binary creates filename.ext_ST instead of filename_ST
        # First try the expected format (filename_ST), then try the actual format (filename.ext_ST)
        st_path = output_path.with_name(f"{output_path.stem}_ST")
        if not st_path.exists():
            # Try the format that includes the original extension
            st_path = output_path.with_suffix(f"{output_path.suffix}_ST")

        # Load the data
        Q = np.loadtxt(st_path)

        # Process according to DDA format: skip first 2 columns and transpose
        if Q.shape[1] > 2:
            print(f"Loaded DDA output shape: {Q.shape}")
            Q = Q[:, 2:]  # Skip first 2 columns
            Q = Q[:, 1::4]  # Take every 4th column starting from column 1 (0-indexed)
            Q = Q.T  # Transpose to get channels × time windows

        return Q, st_path

    def _prepare_execution(
        self,
        input_file: str,
        output_file: Optional[str],
        channel_list: List[int],
        bounds: Optional[Tuple[int, int]],
        cpu_time: bool,
        select_mask: Optional[List[int]],
        model_params: Optional[List[int]],
        delays: Optional[List[int]],
        window_length: Optional[int],
        window_step: Optional[int],
        ct_window_length: Optional[int],
        ct_window_step: Optional[int],
        model_dimension: Optional[int],
        polynomial_order: Optional[int],
        num_tau: Optional[int],
    ) -> Tuple[List[str], Path]:
        """Prepare command and output path for execution."""

        output_path = Path(output_file) if output_file else self._create_tempfile()
        command = self._make_command(
            input_file,
            str(output_path),
            channel_list,
            bounds,
            cpu_time,
            select_mask=select_mask,
            model_params=model_params,
            delays=delays,
            window_length=window_length,
            window_step=window_step,
            ct_window_length=ct_window_length,
            ct_window_step=ct_window_step,
            model_dimension=model_dimension,
            polynomial_order=polynomial_order,
            num_tau=num_tau,
        )

        return command, output_path

    def run(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        channel_list: Optional[List[int]] = None,
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
        raise_on_error: bool = False,
        select_mask: Optional[List[int]] = None,
        model_params: Optional[List[int]] = None,
        delays: Optional[List[int]] = None,
        window_length: Optional[int] = None,
        window_step: Optional[int] = None,
        ct_window_length: Optional[int] = None,
        ct_window_step: Optional[int] = None,
        model_dimension: Optional[int] = None,
        polynomial_order: Optional[int] = None,
        num_tau: Optional[int] = None,
    ) -> Tuple[np.ndarray, Path]:
        """Run DDA synchronously."""

        channels = channel_list if channel_list is not None else []
        command, output_path = self._prepare_execution(
            input_file,
            output_file,
            channels,
            bounds,
            cpu_time,
            select_mask,
            model_params,
            delays,
            window_length,
            window_step,
            ct_window_length,
            ct_window_step,
            model_dimension,
            polynomial_order,
            num_tau,
        )

        # Make binary executable if needed
        if not os.access(self.binary_path, os.X_OK):
            os.chmod(self.binary_path, 0o755)

        # Run APE binary
        process = subprocess.run(command, capture_output=raise_on_error)

        if raise_on_error and process.returncode != 0:
            stderr = process.stderr if process.stderr else b""
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                stderr.decode() if stderr else "Command failed",
            )

        return self._process_output(output_path)

    async def run_async(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        channel_list: Optional[List[int]] = None,
        bounds: Optional[Tuple[int, int]] = None,
        cpu_time: bool = False,
        raise_on_error: bool = False,
        select_mask: Optional[List[int]] = None,
        model_params: Optional[List[int]] = None,
        delays: Optional[List[int]] = None,
        window_length: Optional[int] = None,
        window_step: Optional[int] = None,
        ct_window_length: Optional[int] = None,
        ct_window_step: Optional[int] = None,
        model_dimension: Optional[int] = None,
        polynomial_order: Optional[int] = None,
        num_tau: Optional[int] = None,
    ) -> Tuple[np.ndarray, Path]:
        """Run DDA asynchronously."""

        channels = channel_list if channel_list is not None else []
        command, output_path = self._prepare_execution(
            input_file,
            output_file,
            channels,
            bounds,
            cpu_time,
            select_mask,
            model_params,
            delays,
            window_length,
            window_step,
            ct_window_length,
            ct_window_step,
            model_dimension,
            polynomial_order,
            num_tau,
        )

        # Make binary executable if needed
        if not os.access(self.binary_path, os.X_OK):
            os.chmod(self.binary_path, 0o755)

        # Run APE binary asynchronously
        if raise_on_error:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        await process.wait()

        if raise_on_error and process.returncode != 0:
            stderr = await process.stderr.read() if process.stderr else b""
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                stderr.decode() if stderr else "Command failed",
            )

        return self._process_output(output_path)


# For backward compatibility or simpler usage
def run_dda(*args, **kwargs) -> Tuple[np.ndarray, Path]:
    """Synchronous DDA execution (global instance)."""
    return DDARunner(DDA_BINARY_PATH).run(*args, **kwargs)


async def run_dda_async(*args, **kwargs) -> Tuple[np.ndarray, Path]:
    """Asynchronous DDA execution (global instance)."""
    return await DDARunner(DDA_BINARY_PATH).run_async(*args, **kwargs)
