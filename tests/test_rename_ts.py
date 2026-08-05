"""
Tests for rename-ts's symlink-collision handling.

Was an uncaught FileExistsError raised mid-loop -- a raw traceback instead
of the clean sys.exit(1) every other failure mode in this file uses, and
worse, it could leave orphan symlinks on disk (created before the crash)
with no project.json record of them, since update_section() only runs
after the whole loop completes. Fixed with a pre-flight scan that checks
every target path before creating any symlink.

Synthetic fixtures (no mounted data required) so these always run.
"""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj
from aretomo3_preprocess.commands.rename_ts import run


def _args(input_dir, start=1, digits=None, dry_run=False):
    return argparse.Namespace(input=str(input_dir), start=start, digits=digits,
                              dry_run=dry_run)


def _make_mdocs(in_dir: Path, names):
    for name in names:
        (in_dir / name).write_text('dummy mdoc content\n')


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_basic_symlink_creation(isolated_project):
    in_dir = isolated_project / 'frames'
    in_dir.mkdir()
    _make_mdocs(in_dir, ['Position_1.mdoc', 'Position_2.mdoc'])

    run(_args(in_dir))

    assert (in_dir / 'ts-1.mdoc').is_symlink()
    assert (in_dir / 'ts-2.mdoc').is_symlink()
    grids = pj.load().get('rename_ts', {}).get('grids', {})
    assert grids['1']['n_symlinks'] == 2


def test_collision_exits_cleanly_no_partial_symlinks(isolated_project, capsys):
    in_dir = isolated_project / 'frames'
    in_dir.mkdir()
    _make_mdocs(in_dir, ['Position_1.mdoc', 'Position_2.mdoc', 'Position_3.mdoc'])

    # Pre-create a colliding target for the SECOND planned symlink (ts-2.mdoc)
    # -- simulates a partially-completed prior run at --start 1.
    (in_dir / 'ts-2.mdoc').symlink_to((in_dir / 'Position_2.mdoc').resolve())

    with pytest.raises(SystemExit) as exc_info:
        run(_args(in_dir, start=1))
    assert exc_info.value.code == 1

    # No new symlinks created for ts-1/ts-3 -- the pre-flight check aborts
    # before creating anything, not partway through.
    assert not (in_dir / 'ts-1.mdoc').exists()
    assert not (in_dir / 'ts-3.mdoc').exists()

    # project.json was never touched -- no grid recorded.
    assert pj.load().get('rename_ts', {}) == {}

    err = capsys.readouterr().out
    assert 'ts-2.mdoc' in err
    assert '--start' in err


def test_collision_message_lists_all_conflicts(isolated_project, capsys):
    in_dir = isolated_project / 'frames'
    in_dir.mkdir()
    _make_mdocs(in_dir, ['Position_1.mdoc', 'Position_2.mdoc'])
    (in_dir / 'ts-1.mdoc').symlink_to((in_dir / 'Position_1.mdoc').resolve())
    (in_dir / 'ts-2.mdoc').symlink_to((in_dir / 'Position_2.mdoc').resolve())

    with pytest.raises(SystemExit):
        run(_args(in_dir, start=1))

    out = capsys.readouterr().out
    assert 'ts-1.mdoc' in out
    assert 'ts-2.mdoc' in out
    assert '2 target path(s)' in out


def test_no_collision_past_start_succeeds(isolated_project):
    """Adjusting --start past the existing symlinks (the fix's own advice)
    actually resolves the collision."""
    in_dir = isolated_project / 'frames'
    in_dir.mkdir()
    _make_mdocs(in_dir, ['Position_1.mdoc', 'Position_2.mdoc'])
    (in_dir / 'ts-1.mdoc').symlink_to((in_dir / 'Position_1.mdoc').resolve())

    # Position_1.mdoc is now a symlink target's source but Position_1.mdoc
    # itself is still a real file (not a symlink) so it's still globbed --
    # use a fresh start past ts-1 to avoid the real collision at ts-1.
    run(_args(in_dir, start=5))

    assert (in_dir / 'ts-5.mdoc').is_symlink()
    assert (in_dir / 'ts-6.mdoc').is_symlink()


def test_dry_run_does_not_check_or_create(isolated_project):
    in_dir = isolated_project / 'frames'
    in_dir.mkdir()
    _make_mdocs(in_dir, ['Position_1.mdoc'])
    (in_dir / 'ts-1.mdoc').symlink_to((in_dir / 'Position_1.mdoc').resolve())

    # dry-run must not raise/exit even though ts-1.mdoc already exists --
    # it never touches disk.
    run(_args(in_dir, start=1, dry_run=True))
    assert pj.load().get('rename_ts', {}) == {}
