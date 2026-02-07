"""BIDS dataset integration for DDA analysis."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from .results import CTResult, DEResult, STResult


def _require_mne_bids():
    try:
        import mne_bids

        return mne_bids
    except ImportError:
        raise ImportError(
            "mne-bids is required for BIDS integration. "
            "Install with: pip install 'dda-py[mne-bids]'"
        )


@dataclass
class BIDSRecording:
    """Metadata for a single BIDS recording.

    Attributes:
        subject: Subject ID (e.g. '01').
        session: Session ID or None.
        task: Task name.
        run: Run number or None.
        datatype: Data type (e.g. 'eeg', 'meg').
        fpath: Absolute path to the data file.
    """

    subject: str
    session: Optional[str]
    task: str
    run: Optional[str]
    datatype: str
    fpath: str

    @property
    def label(self) -> str:
        """Human-readable label like 'sub-01_ses-02_task-rest'."""
        parts = [f"sub-{self.subject}"]
        if self.session:
            parts.append(f"ses-{self.session}")
        parts.append(f"task-{self.task}")
        if self.run:
            parts.append(f"run-{self.run}")
        return "_".join(parts)


def find_recordings(
    bids_root: str,
    datatype: str = "eeg",
    task: Optional[str] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    run: Optional[str] = None,
) -> List[BIDSRecording]:
    """Discover BIDS recordings matching filters.

    Args:
        bids_root: Path to BIDS dataset root.
        datatype: Data modality ('eeg', 'meg', 'ieeg').
        task: Filter by task name.
        subject: Filter by subject ID.
        session: Filter by session.
        run: Filter by run.

    Returns:
        List of BIDSRecording objects.
    """
    mne_bids = _require_mne_bids()

    kwargs: Dict[str, Any] = {
        "root": bids_root,
        "datatypes": [datatype],
    }
    if task is not None:
        kwargs["tasks"] = [task]
    if subject is not None:
        kwargs["subjects"] = [subject]
    if session is not None:
        kwargs["sessions"] = [session]
    if run is not None:
        kwargs["runs"] = [run]

    bids_paths = mne_bids.find_matching_paths(**kwargs)

    recordings = []
    for bp in bids_paths:
        recordings.append(
            BIDSRecording(
                subject=bp.subject or "",
                session=bp.session,
                task=bp.task or "",
                run=bp.run,
                datatype=bp.datatype or datatype,
                fpath=str(bp.fpath),
            )
        )

    return recordings


def run_bids(
    bids_root: str,
    variant: str = "st",
    datatype: str = "eeg",
    task: Optional[str] = None,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    delays: Sequence[int] = (7, 10),
    model: Optional[list[int]] = None,
    wl: int = 200,
    ws: int = 100,
    channels: Optional[Union[List[int], List[str]]] = None,
    binary_path: Optional[str] = None,
    n_jobs: int = 1,
    progress: bool = True,
    **kwargs: Any,
) -> "Dict[str, Union[STResult, CTResult, DEResult]]":
    """Run DDA over all matching BIDS recordings.

    Args:
        bids_root: Path to BIDS dataset root.
        variant: 'st', 'ct', or 'de'.
        datatype: Data modality.
        task: Filter by task.
        subject: Filter by subject.
        session: Filter by session.
        delays: Delay values.
        model: Model encoding.
        wl: Window length.
        ws: Window step.
        channels: Channel selection (indices or names).
        binary_path: Path to DDA binary.
        n_jobs: Number of parallel workers.
        progress: Show progress bar.
        **kwargs: Extra args passed to run_st/run_ct/run_de.

    Returns:
        Dict mapping recording label to result object.
    """
    mne_bids = _require_mne_bids()
    import mne

    from .batch import _get_run_func

    recordings = find_recordings(
        bids_root,
        datatype=datatype,
        task=task,
        subject=subject,
        session=session,
    )

    if not recordings:
        return {}

    run_func = _get_run_func(variant)

    run_kwargs: Dict[str, Any] = {
        "delays": delays,
        "wl": wl,
        "ws": ws,
        **kwargs,
    }
    if model is not None:
        run_kwargs["model"] = model
    if channels is not None:
        run_kwargs["channels"] = channels
    if binary_path is not None:
        run_kwargs["binary_path"] = binary_path

    results: Dict[str, Any] = {}
    iterator = enumerate(recordings)

    try:
        from tqdm import tqdm

        if progress:
            iterator = tqdm(
                list(iterator), desc="BIDS DDA", unit="recording"
            )
    except ImportError:
        pass

    for idx, rec in iterator:
        if progress and "tqdm" not in sys.modules:
            print(
                f"Processing {idx + 1}/{len(recordings)}: {rec.label}",
                file=sys.stderr,
            )

        # Read via mne-bids to get MNE Raw object with correct sfreq
        from mne_bids import BIDSPath, read_raw_bids

        bp = BIDSPath(
            subject=rec.subject,
            session=rec.session,
            task=rec.task,
            run=rec.run,
            datatype=rec.datatype,
            root=bids_root,
        )
        raw = read_raw_bids(bids_path=bp, verbose=False)
        raw.load_data()

        result = run_func(raw, **run_kwargs)
        results[rec.label] = result

    return results
