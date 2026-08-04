"""
Tests for the frame_lookup project.json section (SEC <-> acq_order/z_value
bridge, see CLAUDE.md's "frame_lookup" section) and its accessors
(register_frame_lookup, get_frame_lookup, resolve_frame) in
shared/project_state.py.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared import project_json as pj
from aretomo3_preprocess.shared.project_state import (
    register_frame_lookup, get_frame_lookup, resolve_frame,
)

# 6 total frames: SEC 1-5 aligned (acq_order 3,1,5,2,4 -- deliberately not
# tilt-sorted == acq-order to prove sec_to_z isn't just an identity map),
# SEC 6 dark.
_ALIGNED = [  # (sec, acq_order, nominal_tilt)
    (1, 3, -6.0),
    (2, 1,  0.0),
    (3, 5, -12.0),
    (4, 2, -3.0),
    (5, 4, -9.0),
]
_DARK = (6, 6, 6.0)  # sec, acq_order, nominal_tilt


def _write_aln(path: Path, alpha_offset: float = 0.0):
    lines = [
        '# AreTomo Alignment / Priims bprmMn\n',
        '# RawSize = 100 100 6\n',
        '# NumPatches = 0\n',
        f'# AlphaOffset = {alpha_offset:8.2f}\n',
        '# BetaOffset =     0.00\n',
        '# Thickness = 100\n',
        f'# DarkFrame =     0    {_DARK[0]}   {_DARK[2] + alpha_offset:.2f}\n',
        '# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n',
    ]
    for sec, _acq, tilt in _ALIGNED:
        baked = tilt + alpha_offset
        lines.append(f'{sec}  87.29  1.00000  0.0  0.0  1.00  1.00  1.00  0.00  {baked:.2f}\n')
    path.write_text(''.join(lines))


def _write_tlt(path: Path):
    rows = sorted(_ALIGNED + [_DARK], key=lambda r: r[0])
    lines = [f'{tilt:8.2f}   {acq}   3.0000\n' for _sec, acq, tilt in rows]
    path.write_text(''.join(lines))


@pytest.fixture
def synthetic_cmd0_dir(tmp_path, monkeypatch):
    """A tmp cwd (isolated project.json) with ts-001.aln + ts-001_TLT.txt
    written into a subdirectory standing in for a cmd0 output dir."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / 'run001-cmd0'
    out_dir.mkdir()
    _write_aln(out_dir / 'ts-001.aln')
    _write_tlt(out_dir / 'ts-001_TLT.txt')
    return out_dir


def test_register_frame_lookup_basic(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)
    entry = get_frame_lookup('ts-001')
    assert entry is not None
    assert entry['n_total'] == 6
    assert entry['dark_secs'] == [6]
    assert entry['validated'] is True

    expected_z = {str(sec): acq - 1 for sec, acq, _t in _ALIGNED + [_DARK]}
    assert {sec: f['z_value'] for sec, f in entry['frames'].items()} == expected_z
    # No mdoc_data registered in this fixture -- filenames are None, not an error.
    assert all(f['sub_frame_path'] is None for f in entry['frames'].values())


def test_register_frame_lookup_with_alpha_offset_still_validates(tmp_path, monkeypatch):
    """A -TiltCor 1-style .aln (TILT column already shifted by alpha_offset)
    must still validate -- check_nominal_tilt_consistency accounts for it."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / 'run002-cmd1'
    out_dir.mkdir()
    _write_aln(out_dir / 'ts-001.aln', alpha_offset=11.5)
    _write_tlt(out_dir / 'ts-001_TLT.txt')

    register_frame_lookup(out_dir)
    entry = get_frame_lookup('ts-001')
    assert entry['validated'] is True


def test_register_frame_lookup_inconsistent_flags_validated_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / 'run001-cmd0'
    out_dir.mkdir()
    _write_aln(out_dir / 'ts-001.aln')  # alpha_offset=0.0, TILT column raw
    _write_tlt(out_dir / 'ts-001_TLT.txt')

    # Corrupt the .aln's AlphaOffset header AFTER writing raw TILT values,
    # so nominal_tilt + alpha_offset no longer matches TILT for any SEC --
    # simulates a genuinely inconsistent file (e.g. hand-edited badly).
    aln_path = out_dir / 'ts-001.aln'
    text = aln_path.read_text().replace('# AlphaOffset =     0.00', '# AlphaOffset =     7.00')
    aln_path.write_text(text)

    register_frame_lookup(out_dir)
    entry = get_frame_lookup('ts-001')
    assert entry['validated'] is False


def test_get_frame_lookup_unregistered_ts_returns_none(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)
    assert get_frame_lookup('ts-999') is None


def test_resolve_frame_by_sec_z_value_acq_order_agree(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)

    # SEC 3 has acq_order=5 -> z_value=4 (see _ALIGNED).
    by_sec = resolve_frame('ts-001', sec=3)
    by_z = resolve_frame('ts-001', z_value=4)
    by_acq = resolve_frame('ts-001', acq_order=5)

    assert by_sec == by_z == by_acq
    assert by_sec['sec'] == 3
    assert by_sec['z_value'] == 4
    assert by_sec['acq_order'] == 5
    assert by_sec['is_dark'] is False


def test_resolve_frame_flags_dark_sec(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)
    result = resolve_frame('ts-001', sec=6)
    assert result['is_dark'] is True


def test_resolve_frame_unregistered_ts_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_frame('ts-001', sec=1) is None


def test_resolve_frame_unknown_sec_returns_none(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)
    assert resolve_frame('ts-001', sec=999) is None


def test_resolve_frame_requires_exactly_one_identifier(synthetic_cmd0_dir):
    register_frame_lookup(synthetic_cmd0_dir)
    with pytest.raises(ValueError):
        resolve_frame('ts-001')


def test_register_frame_lookup_captures_filename_at_build_time(synthetic_cmd0_dir):
    """Filename is captured when register_frame_lookup() runs (by composing
    with mdoc_data then), not resolved lazily on every resolve_frame() call
    -- so mdoc_data/rename_ts must be registered BEFORE calling
    register_frame_lookup for filenames to be captured."""
    # SEC 2 has acq_order=1 -> z_value=0. Register the rename lookup +
    # mdoc_data matching get_ts_to_original_stem()'s documented
    # rename_ts.lookup shape and mdoc_data.per_ts's original-stem keying.
    pj.update_section('rename_ts', {'lookup': {'ts-001.mdoc': '/frames/Position_1.mdoc'}})
    pj.update_section('mdoc_data', {'per_ts': {
        'Position_1': {'frames': {'0': {'sub_frame_path': 'Position_1_000_0.0.tiff'}}},
    }})

    register_frame_lookup(synthetic_cmd0_dir)

    result = resolve_frame('ts-001', sec=2)
    assert result['z_value'] == 0
    assert result['sub_frame_path'] == 'Position_1_000_0.0.tiff'


def test_resolve_frame_filename_none_when_mdoc_data_missing_at_build_time(synthetic_cmd0_dir):
    """frame_lookup resolves fine even with no mdoc_data/rename_ts at all --
    sub_frame_path is just None, not a failure of the whole lookup."""
    register_frame_lookup(synthetic_cmd0_dir)
    result = resolve_frame('ts-001', sec=2)
    assert result is not None
    assert result['sub_frame_path'] is None


def test_register_frame_lookup_force_refresh_backfills_filename(synthetic_cmd0_dir):
    """Registering mdoc_data AFTER an initial register_frame_lookup call,
    then re-running register_frame_lookup, backfills the filename -- the
    documented recovery path (enrich --frame-lookup --force) for when
    mdoc_data wasn't ready yet the first time."""
    register_frame_lookup(synthetic_cmd0_dir)
    assert resolve_frame('ts-001', sec=2)['sub_frame_path'] is None

    pj.update_section('rename_ts', {'lookup': {'ts-001.mdoc': '/frames/Position_1.mdoc'}})
    pj.update_section('mdoc_data', {'per_ts': {
        'Position_1': {'frames': {'0': {'sub_frame_path': 'Position_1_000_0.0.tiff'}}},
    }})
    register_frame_lookup(synthetic_cmd0_dir)

    assert resolve_frame('ts-001', sec=2)['sub_frame_path'] == 'Position_1_000_0.0.tiff'


def test_resolve_frame_path_none_when_frames_dir_missing(synthetic_cmd0_dir):
    """frames_dir wasn't recorded (mdoc validated before the field existed) --
    frame_path is None, not a lookup failure."""
    pj.update_section('rename_ts', {'lookup': {'ts-001.mdoc': '/frames/Position_1.mdoc'}})
    pj.update_section('mdoc_data', {'per_ts': {
        'Position_1': {'frames': {'0': {'sub_frame_path': 'Position_1_000_0.0.tiff'}}},
    }})
    register_frame_lookup(synthetic_cmd0_dir)

    result = resolve_frame('ts-001', sec=2)
    assert result['sub_frame_path'] == 'Position_1_000_0.0.tiff'
    assert result['frame_path'] is None


def test_resolve_frame_path_joins_frames_dir_with_basename(synthetic_cmd0_dir):
    """frame_path combines frames_dir with sub_frame_path's own FILENAME
    only -- sub_frame_path's directory (a stale Windows/UNC acquisition-PC
    path here) must be discarded, not used."""
    pj.update_section('rename_ts', {'lookup': {'ts-001.mdoc': '/frames/Position_1.mdoc'}})
    pj.update_section('mdoc_data', {'per_ts': {
        'Position_1': {
            'frames': {'0': {
                'sub_frame_path': r'\\ACQPC\staging\Position_1_000_0.0.tiff',
            }},
            'frames_dir': '/frames',
        },
    }})
    register_frame_lookup(synthetic_cmd0_dir)

    result = resolve_frame('ts-001', sec=2)
    assert result['frame_path'] == '/frames/Position_1_000_0.0.tiff'


def test_register_frame_lookup_merges_across_calls(tmp_path, monkeypatch):
    """Calling register_frame_lookup again (e.g. TS processed incrementally)
    adds/refreshes entries rather than dropping previously-registered TS."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / 'run001-cmd0'
    out_dir.mkdir()
    _write_aln(out_dir / 'ts-001.aln')
    _write_tlt(out_dir / 'ts-001_TLT.txt')
    register_frame_lookup(out_dir)

    _write_aln(out_dir / 'ts-002.aln')
    _write_tlt(out_dir / 'ts-002_TLT.txt')
    register_frame_lookup(out_dir)

    assert get_frame_lookup('ts-001') is not None
    assert get_frame_lookup('ts-002') is not None
