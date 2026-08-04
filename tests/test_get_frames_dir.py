"""
Tests for get_frames_dir() (shared/project_state.py).

Was a dead accessor -- it read a top-level rename_ts.input key that
rename_ts.py never writes (the directory is only ever recorded nested at
rename_ts.grids.<N>.input_dir), so it always returned None in real usage.
Fixed to read the actual grids structure. See CLAUDE.md / task history for
the audit finding this came from.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj
from aretomo3_preprocess.shared.project_state import get_frames_dir


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_get_frames_dir_none_when_no_rename_ts(isolated_project):
    assert get_frames_dir() is None


def test_get_frames_dir_single_grid(isolated_project):
    pj.update_section('rename_ts', {
        'grids': {'1': {'input_dir': '/data/session1/frames'}},
    })
    assert get_frames_dir() == Path('/data/session1/frames')


def test_get_frames_dir_multiple_grids_uses_latest(isolated_project, capsys):
    pj.update_section('rename_ts', {
        'grids': {
            '1': {'input_dir': '/data/session1/frames'},
            '2': {'input_dir': '/data/session2/frames'},
        },
    })
    assert get_frames_dir() == Path('/data/session2/frames')
    assert 'WARNING' in capsys.readouterr().out


def test_get_frames_dir_multiple_grids_same_dir_no_warning(isolated_project, capsys):
    """Re-running rename-ts against the same directory (e.g. incremental
    processing) shouldn't warn -- there's no actual ambiguity."""
    pj.update_section('rename_ts', {
        'grids': {
            '1': {'input_dir': '/data/frames'},
            '2': {'input_dir': '/data/frames'},
        },
    })
    assert get_frames_dir() == Path('/data/frames')
    assert 'WARNING' not in capsys.readouterr().out


def test_get_frames_dir_grid_missing_input_dir(isolated_project):
    pj.update_section('rename_ts', {'grids': {'1': {}}})
    assert get_frames_dir() is None
