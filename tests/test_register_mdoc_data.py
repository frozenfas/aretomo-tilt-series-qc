"""
Tests for shared/project_state.py:register_mdoc_data() -- consolidates two
independent implementations that had drifted apart (enrich.py's
_enrich_mdoc_data stripped stale 'ts-\\d+'-keyed entries but didn't record
frames_dir; validate_mdoc.py's _save_mdoc_to_project recorded frames_dir
but didn't strip stale keys). Both now call this single implementation.

Uses a real, minimal synthetic mdoc parsed by the actual mdocfile library
(skipped if not installed) rather than mocking parse_mdoc_file, since the
whole point is testing the merge/key-resolution logic around a real parse.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj
from aretomo3_preprocess.shared.project_state import register_mdoc_data

try:
    import mdocfile  # noqa: F401
    _HAS_MDOCFILE = True
except ImportError:
    _HAS_MDOCFILE = False

skip_if_no_mdocfile = pytest.mark.skipif(not _HAS_MDOCFILE, reason='mdocfile not installed')


def _write_mdoc(path: Path, n_sections=2, stem='Position_1'):
    lines = [
        'DataMode = 6\n',
        'ImageSize = 100 100\n',
        'PixelSpacing = 1.69\n',
        'Voltage = 300.00\n',
        '\n',
    ]
    for i in range(n_sections):
        lines += [
            f'[ZValue = {i}]\n',
            f'TiltAngle = {i * 3.0}\n',
            'ExposureDose = 4.16\n',
            f'SubFramePath = X:\\data\\{stem}_{i:03d}_{i * 3.0}.tiff\n',
            'ExposureTime = 1.0\n',
            'DateTime = 01-Jan-2026  00:00:00\n',
            '\n',
        ]
    path.write_text(''.join(lines))


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@skip_if_no_mdocfile
def test_basic_registration(isolated_project):
    mdoc_path = isolated_project / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, n_sections=2)

    result = register_mdoc_data([mdoc_path])
    assert result == {'n_ok': 1, 'n_fail': 0, 'merged_count': 1}

    per_ts = pj.load()['mdoc_data']['per_ts']
    assert set(per_ts) == {'Position_1'}
    assert len(per_ts['Position_1']['frames']) == 2


@skip_if_no_mdocfile
def test_frames_dir_recorded(isolated_project):
    mdoc_path = isolated_project / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, n_sections=1)

    register_mdoc_data([mdoc_path])
    entry = pj.load()['mdoc_data']['per_ts']['Position_1']
    assert entry['frames_dir'] == str(isolated_project.resolve())


@skip_if_no_mdocfile
def test_symlink_resolves_to_original_stem(isolated_project):
    """A renamed ts-XXX.mdoc symlink must key its entry by the ORIGINAL
    (resolved) stem, purely via filesystem resolution -- no rename_ts.lookup
    project.json dependency needed to get this right."""
    original = isolated_project / 'Position_1.mdoc'
    _write_mdoc(original, n_sections=1)
    symlink = isolated_project / 'ts-001.mdoc'
    symlink.symlink_to(original.resolve())

    result = register_mdoc_data([symlink])
    assert result['n_ok'] == 1
    per_ts = pj.load()['mdoc_data']['per_ts']
    assert set(per_ts) == {'Position_1'}


@skip_if_no_mdocfile
def test_stale_ts_numbered_keys_stripped_on_merge(isolated_project):
    """A bare 'ts-123' key is never a real original stem -- always stale
    debris from an older key-resolution strategy, dropped unconditionally
    before merging in new entries."""
    pj.update_section('mdoc_data', {'per_ts': {
        'ts-999': {'angpix': 1.0, 'acquisition': {}, 'frames': {}},
        'Position_5': {'angpix': 1.0, 'acquisition': {}, 'frames': {}},
    }})

    mdoc_path = isolated_project / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, n_sections=1)
    register_mdoc_data([mdoc_path])

    per_ts = pj.load()['mdoc_data']['per_ts']
    assert 'ts-999' not in per_ts
    assert 'Position_5' in per_ts  # untouched, non-stale entries survive
    assert 'Position_1' in per_ts


@skip_if_no_mdocfile
def test_merge_preserves_other_entries(isolated_project):
    mdoc1 = isolated_project / 'Position_1.mdoc'
    mdoc2 = isolated_project / 'Position_2.mdoc'
    _write_mdoc(mdoc1, n_sections=1, stem='Position_1')
    _write_mdoc(mdoc2, n_sections=1, stem='Position_2')

    register_mdoc_data([mdoc1])
    result = register_mdoc_data([mdoc2])

    assert result['merged_count'] == 2
    per_ts = pj.load()['mdoc_data']['per_ts']
    assert set(per_ts) == {'Position_1', 'Position_2'}


def test_unparseable_file_counted_as_failure(isolated_project):
    bad_path = isolated_project / 'garbage.mdoc'
    bad_path.write_text('not a real mdoc file at all\n')

    result = register_mdoc_data([bad_path])
    assert result['n_ok'] == 0
    assert result['n_fail'] == 1
    assert pj.load().get('mdoc_data', {}) == {}


def test_no_files_writes_nothing(isolated_project):
    result = register_mdoc_data([])
    assert result == {'n_ok': 0, 'n_fail': 0, 'merged_count': 0}
    assert pj.load().get('mdoc_data', {}) == {}
