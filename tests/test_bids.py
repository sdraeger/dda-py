"""Tests for BIDS integration module."""

import pytest

mne_bids = pytest.importorskip("mne_bids")

from dda_py.bids import BIDSRecording, find_recordings


class TestBIDSRecording:
    def test_label_with_session(self):
        rec = BIDSRecording(
            subject="01",
            session="02",
            task="rest",
            run="01",
            datatype="eeg",
            fpath="/data/sub-01/eeg/sub-01_task-rest.edf",
        )
        assert rec.label == "sub-01_ses-02_task-rest_run-01"

    def test_label_without_session(self):
        rec = BIDSRecording(
            subject="01",
            session=None,
            task="rest",
            run=None,
            datatype="eeg",
            fpath="/data/sub-01/eeg/sub-01_task-rest.edf",
        )
        assert rec.label == "sub-01_task-rest"

    def test_label_minimal(self):
        rec = BIDSRecording(
            subject="03",
            session=None,
            task="motor",
            run=None,
            datatype="meg",
            fpath="/path",
        )
        assert rec.label == "sub-03_task-motor"


class TestFindRecordings:
    def test_finds_recordings(self, monkeypatch):
        class MockBIDSPath:
            def __init__(self, subject, session, task, run, datatype, fpath):
                self.subject = subject
                self.session = session
                self.task = task
                self.run = run
                self.datatype = datatype
                self.fpath = fpath

        mock_paths = [
            MockBIDSPath("01", "01", "rest", "01", "eeg", "/data/sub-01.edf"),
            MockBIDSPath("02", "01", "rest", "01", "eeg", "/data/sub-02.edf"),
        ]

        monkeypatch.setattr(
            "dda_py.bids.mne_bids.find_matching_paths",
            lambda **kwargs: mock_paths,
        )

        # Need to also patch _require_mne_bids
        import dda_py.bids as bids_mod

        monkeypatch.setattr(
            bids_mod,
            "_require_mne_bids",
            lambda: __import__("mne_bids"),
        )

        recordings = find_recordings("/data", datatype="eeg")
        assert len(recordings) == 2
        assert recordings[0].subject == "01"
        assert recordings[1].subject == "02"
        assert all(isinstance(r, BIDSRecording) for r in recordings)

    def test_filters_by_task(self, monkeypatch):
        import dda_py.bids as bids_mod

        captured_kwargs = {}

        def mock_find(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        monkeypatch.setattr(
            bids_mod,
            "_require_mne_bids",
            lambda: type("M", (), {"find_matching_paths": mock_find})(),
        )
        monkeypatch.setattr(
            "dda_py.bids.mne_bids.find_matching_paths", mock_find
        )

        find_recordings("/data", task="rest")
        assert captured_kwargs.get("tasks") == ["rest"]
