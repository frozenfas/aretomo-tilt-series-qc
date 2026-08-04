"""
Tests for validate-mdoc's --check-frames option (verifying that
SubFramePath-referenced movies actually exist next to the mdoc, since
AreTomo3 requires raw movies and their mdoc to be co-located -- see
CLAUDE.md's frames/mdoc co-location note and check_frames_found's
docstring in validate_mdoc.py).

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands.validate_mdoc import (
    _frame_basename, check_frames_found, validate_file,
)


def test_frame_basename_strips_windows_unc_path():
    raw = r'\\GATANCUSTOMER\DoseFractions\Supervisor_tomo\Position_1_001_0.0_fractions.tiff'
    assert _frame_basename(raw) == 'Position_1_001_0.0_fractions.tiff'


def test_frame_basename_strips_unix_path():
    raw = '/net/data/acquisition/Position_1_001_0.0_fractions.tiff'
    assert _frame_basename(raw) == 'Position_1_001_0.0_fractions.tiff'


def test_frame_basename_bare_filename_unchanged():
    assert _frame_basename('Position_1_001_0.0_fractions.tiff') == \
        'Position_1_001_0.0_fractions.tiff'


def _section(zval, tilt, fname, dose=4.16, exptime=1.0):
    # The trailing DateTime line matters: _simulate_aretomo3's state machine
    # commits a section only on the line AFTER ExposureTime, whatever it is
    # -- without a filler line here, the next section's own "[ZValue = N]"
    # marker gets silently consumed as that trigger and is never seen as a
    # section start, undercounting n_tilts. Real mdocs always have several
    # more fields (NumSubFrames, DateTime, ...) between ExposureTime and the
    # next section for the same reason.
    return (
        f'[ZValue = {zval}]\n'
        f'TiltAngle = {tilt}\n'
        f'ExposureDose = {dose}\n'
        f'SubFramePath = \\\\ACQPC\\staging\\{fname}\n'
        f'ExposureTime = {exptime}\n'
        f'DateTime = 01-Jan-2026  00:00:00\n'
    )


def _write_mdoc(path: Path, filenames, tilts=None):
    n = len(filenames)
    tilts = tilts if tilts is not None else [float(i) for i in range(n)]
    text = ''.join(_section(i, t, f) for i, (t, f) in enumerate(zip(tilts, filenames)))
    path.write_text(text)


# _MIN_TILTS is 7 -- use 7 sections so these mdocs are otherwise fully valid.
_FILENAMES = [f'Position_1_{i:03d}_0.0_fractions.tiff' for i in range(7)]


def test_check_frames_found_all_present(tmp_path):
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    for fname in _FILENAMES:
        (tmp_path / fname).write_bytes(b'')

    lines = [l + '\n' for l in mdoc_path.read_text().splitlines()]
    result = check_frames_found(mdoc_path, lines)
    assert result == {'n_frames': 7, 'n_found': 7, 'missing': []}


def test_check_frames_found_some_missing(tmp_path):
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    # Only create the first 5 of 7 referenced movies.
    for fname in _FILENAMES[:5]:
        (tmp_path / fname).write_bytes(b'')

    lines = [l + '\n' for l in mdoc_path.read_text().splitlines()]
    result = check_frames_found(mdoc_path, lines)
    assert result['n_frames'] == 7
    assert result['n_found'] == 5
    assert set(result['missing']) == set(_FILENAMES[5:])


def test_validate_file_check_frames_false_ignores_missing_movies(tmp_path):
    """Default (check_frames=False) behavior is unchanged -- an mdoc with no
    movies on disk at all still passes if its fields are otherwise valid."""
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    r = validate_file(str(mdoc_path))
    assert r['success'] is True
    assert r['n_frames_expected'] is None
    assert r['n_frames_found'] is None


def test_validate_file_check_frames_true_fails_on_missing_movies(tmp_path):
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    # No movie files created at all.
    r = validate_file(str(mdoc_path), check_frames=True)
    assert r['success'] is False
    assert r['n_frames_expected'] == 7
    assert r['n_frames_found'] == 0
    assert any('not found' in issue for issue in r['issues'])


def test_validate_file_check_frames_true_passes_when_movies_present(tmp_path):
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    for fname in _FILENAMES:
        (tmp_path / fname).write_bytes(b'')

    r = validate_file(str(mdoc_path), check_frames=True)
    assert r['success'] is True
    assert r['n_frames_expected'] == 7
    assert r['n_frames_found'] == 7


def test_validate_file_check_frames_true_movies_in_wrong_dir_still_fail(tmp_path):
    """SubFramePath's own (stale acquisition-PC) directory is never used for
    the existence check -- only mdoc_path's actual parent. Dropping the
    movies in an unrelated directory must still fail."""
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_mdoc(mdoc_path, _FILENAMES)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    for fname in _FILENAMES:
        (elsewhere / fname).write_bytes(b'')

    r = validate_file(str(mdoc_path), check_frames=True)
    assert r['success'] is False
    assert r['n_frames_found'] == 0


def test_validate_file_fix_dose_does_not_override_missing_frames(tmp_path):
    """A --fix-dose run that successfully injects ExposureDose must not
    report success=True if movies are still missing -- injecting a dose
    value doesn't make the referenced movies exist."""
    def _section_no_dose(zval, tilt, fname):
        return (
            f'[ZValue = {zval}]\n'
            f'TiltAngle = {tilt}\n'
            f'SubFramePath = \\\\ACQPC\\staging\\{fname}\n'
            f'ExposureTime = 1.0\n'
            f'DateTime = 01-Jan-2026  00:00:00\n'
        )
    text = ''.join(_section_no_dose(i, float(i), f) for i, f in enumerate(_FILENAMES))
    mdoc_path = tmp_path / 'Position_1.mdoc'
    mdoc_path.write_text(text)
    # No movie files present at all.

    r = validate_file(str(mdoc_path), fix_dose=True, dose=4.16, check_frames=True)
    assert r['fixed'] is True
    assert 'dose' in r['fix_types']
    assert r['success'] is False
    assert r['n_frames_found'] == 0
