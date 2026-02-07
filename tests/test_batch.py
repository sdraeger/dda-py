"""Tests for batch processing module."""

import numpy as np
import pytest

from dda_py.batch import GroupResult, collect_results, run_batch


class TestCollectResults:
    def test_st_results(self, mock_st_result):
        results = [mock_st_result(seed=i) for i in range(5)]
        group = collect_results(results)

        assert isinstance(group, GroupResult)
        assert group.n_subjects == 5
        assert group.n_channels == 3
        assert group.n_windows == 10
        assert group.n_coeffs == 3
        assert group.variant == "ST"
        assert len(group.subject_labels) == 5

    def test_ct_results(self, mock_ct_result):
        results = [mock_ct_result(seed=i) for i in range(3)]
        group = collect_results(results)

        assert group.variant == "CT"
        assert group.n_subjects == 3
        assert group.n_channels == 3  # 3 pairs

    def test_de_results(self, mock_de_result):
        results = [mock_de_result(seed=i) for i in range(4)]
        group = collect_results(results)

        assert group.variant == "DE"
        assert group.n_subjects == 4
        assert group.n_channels == 1
        assert group.n_coeffs == 1
        assert group.channel_labels == ["ergodicity"]

    def test_mismatched_types_raises(self, mock_st_result, mock_ct_result):
        with pytest.raises(ValueError, match="same type"):
            collect_results([mock_st_result(), mock_ct_result()])

    def test_mismatched_channels_raises(self, mock_st_result):
        results = [mock_st_result(n_ch=3), mock_st_result(n_ch=4, seed=1)]
        with pytest.raises(ValueError, match="Channel count"):
            collect_results(results)

    def test_truncation_to_min_windows(self, mock_st_result):
        results = [
            mock_st_result(n_win=10, seed=0),
            mock_st_result(n_win=5, seed=1),
        ]
        with pytest.warns(UserWarning, match="Truncating"):
            group = collect_results(results)
        assert group.n_windows == 5

    def test_custom_labels(self, mock_st_result):
        results = [mock_st_result(seed=i) for i in range(3)]
        group = collect_results(results, labels=["patient1", "patient2", "patient3"])
        assert group.subject_labels == ["patient1", "patient2", "patient3"]

    def test_mean_over_windows(self, mock_st_result):
        results = [mock_st_result(seed=i) for i in range(3)]
        group = collect_results(results)
        mean = group.mean_over_windows()
        assert mean.shape == (3, 3, 3)  # (n_subjects, n_channels, n_coeffs)

    def test_to_dataframe(self, mock_st_result):
        pd = pytest.importorskip("pandas")
        results = [mock_st_result(n_win=3, seed=i) for i in range(2)]
        group = collect_results(results)
        df = group.to_dataframe()

        assert len(df) == 2 * 3 * 3  # subjects * channels * windows
        assert "subject" in df.columns
        assert "channel" in df.columns
        assert "a_1" in df.columns

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            collect_results([])


class TestRunBatch:
    def test_dispatches_st(self, monkeypatch, tmp_path, mock_st_result):
        # Create dummy files
        for i in range(3):
            data = np.random.randn(100, 2)  # 100 samples, 2 channels
            np.savetxt(tmp_path / f"file{i}.txt", data)

        files = [str(tmp_path / f"file{i}.txt") for i in range(3)]

        # Mock run_st to return mock results
        mock_result = mock_st_result(n_ch=2)
        monkeypatch.setattr(
            "dda_py.batch._get_run_func",
            lambda v: lambda data, **kw: mock_result,
        )

        results = run_batch(files, variant="st", sfreq=256.0, progress=False)
        assert len(results) == 3

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            run_batch([], variant="invalid")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            run_batch(["/nonexistent/file.txt"], variant="st")

    def test_progress_fallback(self, monkeypatch, tmp_path, mock_st_result, capsys):
        data = np.random.randn(100, 2)
        np.savetxt(tmp_path / "file.txt", data)

        mock_result = mock_st_result(n_ch=2)
        monkeypatch.setattr(
            "dda_py.batch._get_run_func",
            lambda v: lambda data, **kw: mock_result,
        )

        # Force tqdm import to fail
        import dda_py.batch as batch_mod

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        monkeypatch.delitem(
            __import__("sys").modules, "tqdm", raising=False
        )

        results = run_batch(
            [str(tmp_path / "file.txt")],
            variant="st",
            sfreq=256.0,
            progress=True,
        )
        assert len(results) == 1
