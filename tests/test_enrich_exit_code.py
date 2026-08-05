"""
Tests for enrich.py's exit code on a silent no-op.

Audit finding: all five handlers failed silently/non-fatally on an empty
glob, with did_anything=True set regardless of whether a handler
actually found anything to register. `enrich --mdoc-data empty_dir/`
printed an error line but exited 0, giving a scripted pipeline chain no
signal. Fixed by having each handler return whether it actually
registered something, and run() tracking that separately from
"was a section requested at all".

Synthetic fixtures (no mounted data required) so these always run.
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj
from aretomo3_preprocess.commands.enrich import run


def _args(**overrides):
    base = dict(
        mdoc_data=None, mrc_data=None, tlt_data=None, frame_lookup=None,
        lamellae=None, force=False, in_skips=['_CTF', '_Vol', '_EVN', '_ODD'],
        set_path_aretomo3=None, set_path_pytom=None, set_path_imod=None,
        set_path_gapstop=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_empty_mrc_data_dir_exits_1(isolated_project):
    """The exact scenario the audit flagged, using --mrc-data (no
    mdocfile dependency needed to reproduce it)."""
    empty_dir = isolated_project / 'empty'
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        run(_args(mrc_data=str(empty_dir)))
    assert exc_info.value.code == 1


def test_empty_tlt_data_dir_exits_1(isolated_project):
    empty_dir = isolated_project / 'empty'
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        run(_args(tlt_data=str(empty_dir)))
    assert exc_info.value.code == 1


def test_empty_frame_lookup_dir_exits_1(isolated_project):
    empty_dir = isolated_project / 'empty'
    empty_dir.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        run(_args(frame_lookup=str(empty_dir)))
    assert exc_info.value.code == 1


def test_lamellae_csv_with_no_valid_rows_exits_1(isolated_project):
    csv_path = isolated_project / 'lamella_positions.csv'
    csv_path.write_text('ts_name,lamella\n')  # header only, no data rows
    with pytest.raises(SystemExit) as exc_info:
        run(_args(lamellae=str(csv_path)))
    assert exc_info.value.code == 1


def test_successful_mrc_data_does_not_exit(isolated_project):
    mrc_dir = isolated_project / 'run001'
    mrc_dir.mkdir()
    (mrc_dir / 'ts-001.mrc').write_bytes(b'\x00' * 1024)
    run(_args(mrc_data=str(mrc_dir)))  # must not raise SystemExit
    assert pj.load()['input_stacks']['n_stacks'] == 1


def test_one_success_one_failure_still_exits_1(isolated_project):
    """A mix of a successful section and a failed one must still signal
    failure overall -- not let the successful one mask the other."""
    mrc_dir = isolated_project / 'run001'
    mrc_dir.mkdir()
    (mrc_dir / 'ts-001.mrc').write_bytes(b'\x00' * 1024)
    empty_dir = isolated_project / 'empty'
    empty_dir.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        run(_args(mrc_data=str(mrc_dir), tlt_data=str(empty_dir)))
    assert exc_info.value.code == 1
    # The successful section still actually registered its data.
    assert pj.load()['input_stacks']['n_stacks'] == 1


def test_set_path_alone_does_not_exit(isolated_project):
    run(_args(set_path_aretomo3='/opt/AreTomo3/AreTomo3'))  # must not raise
    assert pj.load()['tool_paths']['aretomo3'] == '/opt/AreTomo3/AreTomo3'


def test_no_args_at_all_exits_1(isolated_project):
    with pytest.raises(SystemExit) as exc_info:
        run(_args())
    assert exc_info.value.code == 1


def test_already_registered_without_force_does_not_count_as_failure(isolated_project):
    """Skipping because data is already registered (no --force) is a
    benign, expected outcome for an idempotent re-run -- not a failure."""
    mrc_dir = isolated_project / 'run001'
    mrc_dir.mkdir()
    (mrc_dir / 'ts-001.mrc').write_bytes(b'\x00' * 1024)
    run(_args(mrc_data=str(mrc_dir)))  # first run: registers

    run(_args(mrc_data=str(mrc_dir)))  # second run: skipped, still not an error
