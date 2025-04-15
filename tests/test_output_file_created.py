import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import asyncio
import subprocess
import tempfile
import importlib

import numpy as np

import dda_py
from dda_py import run_dda, run_dda_async, init, DDARunner


class TestDDARunner(unittest.TestCase):
    def setUp(self):
        self.channels_indices = ["1", "2", "3"]
        self.input_file = "/cnl/xaos/JULIA/EPILEPSY/S04__05_02.edf"
        self.output_file = "tests/data/test_output.txt"
        self.dda_binary_path = "/home/claudia/TOOLS/run_DDA_ASCII"
        self.bounds = (10, 20)
        dda_py.DDA_BINARY_PATH = None

    # --- Init and Setup Tests ---
    def test_init_raises_error_if_dda_binary_not_found(self):
        """Test that init raises FileNotFoundError for a non-existent binary path."""

        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                init("tests/bin/run_DDA_EPILEPSY_NOT_FOUND")

    def test_init_sets_dda_binary_path(self):
        """Test that init correctly sets and returns the DDA binary path."""

        # Ensure Path.exists() returns True for this specific path
        with patch("pathlib.Path.exists", return_value=True) as mock_exists:
            result = init(self.dda_binary_path)
            self.assertEqual(result, self.dda_binary_path)

            # Force reload or re-import to ensure we get the updated module state
            importlib.reload(dda_py)

            self.assertEqual(dda_py.DDA_BINARY_PATH, self.dda_binary_path)
            mock_exists.assert_called_once()

    def test_ddarunner_raises_if_binary_path_not_provided(self):
        """Test that DDARunner raises ValueError if binary path is not set."""
        with self.assertRaises(ValueError):
            DDARunner()

    # --- Helper Method Tests ---
    @patch("tempfile.NamedTemporaryFile")
    def test_create_tempfile_generates_path(self, mock_tempfile):
        """Test _create_tempfile creates a temp file in the .dda directory."""

        # Configure the mock to return a realistic temp file path via its name attribute
        temp_path = f"{tempfile.gettempdir()}/.dda/test/testfile123"
        mock_file = MagicMock()
        mock_file.name = temp_path
        mock_tempfile.return_value = mock_file

        # Call the method
        result = DDARunner._create_tempfile(subdir="test")

        # Assert the result is a Path object starting with the expected directory
        expected_prefix = str(Path(tempfile.gettempdir()) / ".dda" / "test")
        self.assertTrue(str(result).startswith(expected_prefix))
        mock_tempfile.assert_called_once_with(
            dir=Path(tempfile.gettempdir()) / ".dda" / "test", delete=False
        )

    def test_make_command_basic(self):
        """Test _make_command constructs a basic command correctly."""

        init(self.dda_binary_path)

        command = DDARunner._make_command(
            self.input_file, self.output_file, self.channels_indices
        )
        expected = [
            self.dda_binary_path,
            "-DATA_FN",
            self.input_file,
            "-OUT_FN",
            self.output_file,
            "-EDF",
            "-CH_list",
            *self.channels_indices,
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

        self.assertEqual(command, expected)

    def test_make_command_with_bounds_and_cpu_time(self):
        """Test _make_command includes bounds and CPU time when provided."""

        init(self.dda_binary_path)

        command = DDARunner._make_command(
            self.input_file, self.output_file, self.channels_indices, self.bounds, True
        )
        self.assertIn("-StartEnd", command)
        self.assertIn("10", command)
        self.assertIn("20", command)
        self.assertIn("-CPUtime", command)

    @patch("pathlib.Path.read_text", return_value="line1\nline2\nline3\n")
    @patch("pathlib.Path.write_text")
    @patch("numpy.loadtxt", return_value=np.array([[1, 2], [3, 4]]))
    def test_process_output(self, mock_loadtxt, mock_write_text, mock_read_text):
        """Test _process_output processes the ST file correctly."""

        output_path = Path("test_output.txt")
        Q, st_path = DDARunner._process_output(output_path)

        self.assertEqual(st_path, Path("test_output_ST"))

        mock_read_text.assert_called_once()
        mock_write_text.assert_called_once_with("line1\nline2")

        self.assertTrue(np.array_equal(Q, np.array([[1, 2], [3, 4]])))

    # --- Synchronous Execution Tests ---
    @patch("subprocess.run")
    def test_run_creates_output_file(self, mock_subprocess_run):
        """Test run creates an output file when none is provided."""

        mock_subprocess_run.return_value = MagicMock(returncode=0)
        with patch.object(
            DDARunner, "_process_output", return_value=(np.array([]), Path("test_ST"))
        ):
            with patch.object(
                DDARunner, "_create_tempfile", return_value=Path("temp_output")
            ):
                runner = DDARunner(self.dda_binary_path)
                _, st_path = runner.run(self.input_file, None, self.channels_indices)
                self.assertEqual(st_path, Path("test_ST"))
                mock_subprocess_run.assert_called_once()

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["cmd"]))
    def test_run_raises_on_subprocess_error(self, mock_subprocess_run):
        """Test run raises an error if subprocess fails."""

        runner = DDARunner(self.dda_binary_path)
        with self.assertRaises(subprocess.CalledProcessError):
            runner.run(self.input_file, self.output_file, self.channels_indices)

    # --- Asynchronous Execution Tests ---
    @patch("asyncio.create_subprocess_exec")
    async def run_async_creates_output_file(self, mock_subprocess_exec):
        """Helper to test run_async creates an output file when none is provided."""

        mock_process = MagicMock()

        async def wait_mock():
            pass

        mock_process.wait = wait_mock
        mock_process.returncode = 0
        mock_subprocess_exec.return_value = mock_process

        with patch.object(
            DDARunner, "_process_output", return_value=(np.array([]), Path("test_ST"))
        ):
            with patch.object(
                DDARunner, "_create_tempfile", return_value=Path("temp_output")
            ):
                runner = DDARunner(self.dda_binary_path)
                _, st_path = await runner.run_async(
                    self.input_file, None, self.channels_indices
                )
                self.assertEqual(st_path, Path("test_ST"))
                mock_subprocess_exec.assert_called_once()

    @patch("asyncio.create_subprocess_exec")
    async def run_async_raises_on_subprocess_error(self, mock_subprocess_exec):
        """Helper to test run_async raises an error if subprocess fails."""

        mock_process = MagicMock()

        async def wait_mock():
            pass

        async def read_mock():
            return b"error"

        mock_process.wait = wait_mock
        mock_process.stderr.read = read_mock
        mock_process.returncode = 1
        mock_subprocess_exec.return_value = mock_process
        runner = DDARunner(self.dda_binary_path)

        with self.assertRaises(subprocess.CalledProcessError):
            await runner.run_async(
                self.input_file,
                self.output_file,
                self.channels_indices,
                raise_on_error=True,
            )

    async def run_dda_async_wrapper(self):
        """Helper to test run_dda_async wrapper works with initialized binary path."""

        init(self.dda_binary_path)

        with patch.object(
            DDARunner, "run_async", return_value=(np.array([]), Path("test_ST"))
        ):
            _, st_path = await run_dda_async(
                self.input_file, None, self.channels_indices
            )
            self.assertEqual(st_path, Path("test_ST"))

    def test_async_methods(self):
        """Test all async methods using asyncio.run."""
        asyncio.run(self.run_async_creates_output_file())
        asyncio.run(self.run_async_raises_on_subprocess_error())
        asyncio.run(self.run_dda_async_wrapper())

    # --- Function Wrapper Tests ---
    def test_run_dda_wrapper(self):
        """Test run_dda wrapper works with initialized binary path."""

        init(self.dda_binary_path)

        with patch.object(
            DDARunner, "run", return_value=(np.array([]), Path("test_ST"))
        ):
            _, st_path = run_dda(self.input_file, None, self.channels_indices)
            self.assertEqual(st_path, Path("test_ST"))

    # Run async tests using asyncio.run
    def test_run_async_wrapper(self):
        asyncio.run(self.run_async_creates_output_file())
        asyncio.run(self.run_async_raises_on_subprocess_error())
        asyncio.run(self.run_dda_async_wrapper())


if __name__ == "__main__":
    unittest.main()
