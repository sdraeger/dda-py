"""Tests for high-level API functions.

Integration tests are marked with @pytest.mark.integration and require
the DDA binary to be available on the system.
"""

import numpy as np
import pytest

from dda_py.api import _extract_data, _write_temp_ascii
from dda_py.results import STResult, CTResult, DEResult


class TestExtractData:
    """Test _extract_data helper."""

    def test_2d_array(self):
        data = np.random.randn(3, 1000)
        arr, sfreq, labels = _extract_data(data, sfreq=256.0)
        assert arr.shape == (3, 1000)
        assert sfreq == 256.0
        assert labels == ["ch0", "ch1", "ch2"]

    def test_1d_array_reshaped(self):
        data = np.random.randn(1000)
        arr, sfreq, labels = _extract_data(data, sfreq=100.0)
        assert arr.shape == (1, 1000)
        assert labels == ["ch0"]

    def test_channel_selection(self):
        data = np.random.randn(5, 1000)
        arr, sfreq, labels = _extract_data(data, sfreq=256.0, channels=[1, 3])
        assert arr.shape == (2, 1000)
        assert labels == ["ch1", "ch3"]

    def test_3d_array_raises(self):
        data = np.random.randn(2, 3, 1000)
        with pytest.raises(ValueError, match="1D or 2D"):
            _extract_data(data, sfreq=256.0)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="np.ndarray"):
            _extract_data([[1, 2, 3]], sfreq=256.0)

    def test_string_channels_without_mne_raises(self):
        data = np.random.randn(3, 1000)
        with pytest.raises(ValueError, match="String channel names require MNE"):
            _extract_data(data, sfreq=256.0, channels=["Fp1", "Fp2"])


class TestWriteTempAscii:
    """Test _write_temp_ascii helper."""

    def test_writes_transposed(self):
        import os
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
        path = _write_temp_ascii(data)
        try:
            loaded = np.loadtxt(path)
            # data.T should be (3, 2) — 3 rows, 2 columns
            assert loaded.shape == (3, 2)
            np.testing.assert_allclose(loaded, data.T, atol=1e-8)
        finally:
            os.unlink(path)

    def test_temp_file_suffix(self):
        import os
        data = np.random.randn(1, 100)
        path = _write_temp_ascii(data)
        try:
            assert path.endswith(".txt")
            assert "dda_input_" in path
        finally:
            os.unlink(path)


@pytest.mark.integration
class TestRunST:
    """Integration tests for run_st (require DDA binary)."""

    def test_basic_run(self):
        from dda_py import run_st
        data = np.random.randn(2, 2000)
        result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
        assert isinstance(result, STResult)
        assert result.n_channels == 2
        assert result.n_windows > 0
        assert result.coefficients.shape[0] == 2

    def test_single_channel(self):
        from dda_py import run_st
        data = np.random.randn(1, 2000)
        result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
        assert result.n_channels == 1

    def test_1d_input(self):
        from dda_py import run_st
        data = np.random.randn(2000)
        result = run_st(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
        assert result.n_channels == 1


@pytest.mark.integration
class TestRunCT:
    """Integration tests for run_ct (require DDA binary)."""

    def test_basic_run(self):
        from dda_py import run_ct
        data = np.random.randn(3, 2000)
        result = run_ct(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
        assert isinstance(result, CTResult)
        assert result.n_pairs == 3  # 3 choose 2

    def test_too_few_channels(self):
        from dda_py import run_ct
        data = np.random.randn(1, 2000)
        with pytest.raises(ValueError, match="at least 2 channels"):
            run_ct(data, sfreq=256.0)


@pytest.mark.integration
class TestRunDE:
    """Integration tests for run_de (require DDA binary)."""

    def test_basic_run(self):
        from dda_py import run_de
        data = np.random.randn(2, 2000)
        result = run_de(data, sfreq=256.0, delays=(7, 10), wl=200, ws=100)
        assert isinstance(result, DEResult)
        assert result.n_windows > 0
