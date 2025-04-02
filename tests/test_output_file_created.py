import os
import unittest

from dda_py import run_dda, init


class TestFileOps(unittest.TestCase):
    def setUp(self):
        self.channels_indices = ["1", "2", "3"]
        self.input_file = "/cnl/xaos/JULIA/EPILEPSY/S04__05_02.edf"
        self.output_file = "tests/data/test_output.txt"
        self.dda_binary_path = "/home/claudia/TOOLS/run_DDA_ASCII"

    def test_init_raises_error_if_dda_binary_not_found(self):
        with self.assertRaises(FileNotFoundError):
            init("tests/bin/run_DDA_EPILEPSY_NOT_FOUND")

    def test_init_sets_dda_binary_path(self):
        dda_binary_path = init(self.dda_binary_path)
        self.assertEqual(dda_binary_path, self.dda_binary_path)

    def test_output_file_created_if_not_provided(self):
        init(self.dda_binary_path)
        _, output_file = run_dda(
            self.input_file, self.output_file, self.channels_indices
        )
        self.assertTrue(os.path.exists(output_file))


if __name__ == "__main__":
    unittest.main()
