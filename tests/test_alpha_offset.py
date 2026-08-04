"""
Tests for the alpha_offset handling convention, corrected 2026-08 after a
dataset-wide test (see CLAUDE.md's alpha_offset convention section) showed
the v2 assumption below was wrong: AreTomo3's -TiltCor 1 DOES bake
AlphaOffset directly into the .aln TILT column (and the IMOD _st.tlt file
derived from it) -- confirmed on all 172 TS of a real project, comparing
the same TS processed with -TiltCor 0 (raw nominal, matches source mdoc
exactly) vs -TiltCor 1 (every TILT value shifted by precisely that TS's own
AlphaOffset). AlphaOffset is only genuinely still "owed" on top of TILT
when read from AreTomo3's own _TLT.txt (always raw nominal regardless of
TiltCor) -- consumers reading the TILT column / IMOD _st.tlt directly
(pytom_match.py, gapstop_match.py, relion5_convert.py's primary path) must
NOT add it again. aln_edit.py now bakes its offset into TILT (and the
matching IMOD _st.tlt) too, to keep this rule universally true regardless
of whether an .aln came straight from AreTomo3 or was hand-corrected.

These use synthetic fixtures (no mounted data required) so they always run.
"""
import sys
import importlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands import pytom_match, gapstop_match, aln_edit
from aretomo3_preprocess.commands.validate_mdoc import check_parser_conformance


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic .aln / _TLT.txt builders
# ─────────────────────────────────────────────────────────────────────────────

# Five tilts, tilt-sorted, sec 1..5, nominal tilts -6..+6 in steps of 3.
_NOMINAL_TILTS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def _write_aln(path: Path, alpha_offset: float):
    """Matches real AreTomo3 -TiltCor 1 output: the TILT column already has
    alpha_offset baked in (not just recorded in the header) -- see module
    docstring. alpha_offset=0.0 reproduces -TiltCor 0 (raw nominal, header
    and TILT column both untouched)."""
    lines = [
        '# AreTomo Alignment / Priims bprmMn\n',
        '# RawSize = 100 100 5\n',
        '# NumPatches = 0\n',
        f'# AlphaOffset = {alpha_offset:8.2f}\n',
        '# BetaOffset =     0.00\n',
        '# Thickness = 100\n',
        '# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n',
    ]
    for i, tilt in enumerate(_NOMINAL_TILTS, start=1):
        baked_tilt = tilt + alpha_offset
        lines.append(f'{i}  87.29  1.00000  0.0  0.0  1.00  1.00  1.00  0.00  {baked_tilt:.2f}\n')
    path.write_text(''.join(lines))


def _write_tlt(path: Path):
    """AreTomo3's own _TLT.txt format (nominal_tilt, acq_order, dose) --
    always raw nominal regardless of TiltCor, unlike .aln's TILT column."""
    lines = []
    for i, tilt in enumerate(_NOMINAL_TILTS, start=1):
        lines.append(f'{tilt:8.2f}   {i}   1.0000\n')
    path.write_text(''.join(lines))


def _write_tlt_plain(path: Path):
    """IMOD-format .tlt (ts-XXX_Imod/ts-XXX_st.tlt): one tilt angle per
    line, no other columns."""
    path.write_text(''.join(f'{t:8.2f}\n' for t in _NOMINAL_TILTS))


@pytest.fixture(params=[0.0, -14.0], ids=['alpha=0 (v1 parity)', 'alpha=-14 (v2 correction)'])
def synthetic_ts(tmp_path, request):
    """A synthetic ts-001.aln + ts-001_TLT.txt pair with the given alpha_offset."""
    alpha = request.param
    _write_aln(tmp_path / 'ts-001.aln', alpha)
    _write_tlt(tmp_path / 'ts-001_TLT.txt')
    return tmp_path, alpha


# ─────────────────────────────────────────────────────────────────────────────
# pytom_match.py / gapstop_match.py: use the .aln TILT column directly --
# it already has alpha_offset baked in (via -TiltCor 1), never add it again
# ─────────────────────────────────────────────────────────────────────────────

def test_pytom_match_uses_tilt_column_directly(synthetic_ts):
    aretomo_dir, alpha = synthetic_ts
    tlt_out, defocus_out, exposure_out = pytom_match._read_ts_metadata(
        aretomo_dir, 'ts-001')
    # _write_aln already bakes alpha into the TILT column (matching real
    # -TiltCor 1 output) -- pytom_match must use it as-is, not add alpha
    # again, so the expected result is the same baked-in value either way.
    expected = [t + alpha for t in _NOMINAL_TILTS]
    assert tlt_out == pytest.approx(expected, abs=1e-6)
    if alpha == 0.0:
        assert tlt_out == pytest.approx(_NOMINAL_TILTS, abs=1e-6)


def test_gapstop_match_uses_tilt_column_directly(synthetic_ts):
    aretomo_dir, alpha = synthetic_ts
    tilt_angles, defocus_df, exposure_arr, frames = gapstop_match._read_ts_metadata(
        aretomo_dir, 'ts-001')
    expected = [t + alpha for t in _NOMINAL_TILTS]
    assert list(tilt_angles) == pytest.approx(expected, abs=1e-6)
    if alpha == 0.0:
        assert list(tilt_angles) == pytest.approx(_NOMINAL_TILTS, abs=1e-6)


def test_pytom_match_and_gapstop_match_agree(synthetic_ts):
    """Both consumers must derive the same corrected tilt from the same .aln."""
    aretomo_dir, alpha = synthetic_ts
    tlt_out, _, _ = pytom_match._read_ts_metadata(aretomo_dir, 'ts-001')
    tilt_angles, _, _, _ = gapstop_match._read_ts_metadata(aretomo_dir, 'ts-001')
    assert tlt_out == pytest.approx(list(tilt_angles), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# aln-edit: bakes the offset into BOTH the AlphaOffset header AND the TILT
# column (matching AreTomo3's own -TiltCor 1 behavior), plus the matching
# IMOD _st.tlt file if present
# ─────────────────────────────────────────────────────────────────────────────

def test_aln_edit_bakes_offset_into_tilt_column(tmp_path, monkeypatch):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = tmp_path / 'ts-001.aln'
    _write_aln(aln_path, alpha_offset=0.0)

    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)

    new_text = aln_path.read_text()
    new_tilts = [float(l.split()[-1]) for l in new_text.splitlines()
                 if l and not l.startswith('#')]
    assert new_tilts == pytest.approx([t - 14.0 for t in _NOMINAL_TILTS], abs=1e-6), \
        'aln-edit must bake the offset into the TILT column, matching -TiltCor 1'

    assert '# AlphaOffset =  -14.00' in new_text or '# AlphaOffset =   -14.00' in new_text

    # backup preserves the true original (alpha=0, TILT column unshifted)
    bak_path = aln_path.with_suffix(aln_path.suffix + '.bak')
    assert bak_path.exists()
    bak_text = bak_path.read_text()
    assert '# AlphaOffset =     0.00' in bak_text
    bak_tilts = [float(l.split()[-1]) for l in bak_text.splitlines()
                 if l and not l.startswith('#')]
    assert bak_tilts == pytest.approx(_NOMINAL_TILTS, abs=1e-6)


def test_aln_edit_updates_matching_imod_tlt(tmp_path, monkeypatch):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = tmp_path / 'ts-001.aln'
    _write_aln(aln_path, alpha_offset=0.0)
    imod_dir = tmp_path / 'ts-001_Imod'
    imod_dir.mkdir()
    imod_tlt_path = imod_dir / 'ts-001_st.tlt'
    _write_tlt_plain(imod_tlt_path)

    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)

    new_tilts = [float(l) for l in imod_tlt_path.read_text().splitlines() if l.strip()]
    assert new_tilts == pytest.approx([t - 14.0 for t in _NOMINAL_TILTS], abs=1e-6), \
        'aln-edit must apply the same offset to the matching IMOD _st.tlt file'
    assert imod_tlt_path.with_suffix('.tlt.bak').exists()


def test_aln_edit_no_compounding_on_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = tmp_path / 'ts-001.aln'
    _write_aln(aln_path, alpha_offset=0.0)

    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)
    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)  # run again

    text = aln_path.read_text()
    assert '-14.00' in text
    assert '-28.00' not in text, 'repeated aln-edit runs must not compound the offset'


# ─────────────────────────────────────────────────────────────────────────────
# validate_mdoc: AreTomo3 <2.3.0 vs >=2.3.0 conformance
# ─────────────────────────────────────────────────────────────────────────────

def _mdoc_section(z, tilt, dose, path, exptime=None, order='normal'):
    lines = [f'[ZValue = {z}]\n', f'TiltAngle = {tilt}\n', f'ExposureDose = {dose}\n']
    if exptime is not None and order == 'before_path':
        lines.append(f'ExposureTime = {exptime}\n')
    lines.append(f'SubFramePath = X:\\{path}.tif\n')
    if exptime is not None and order == 'normal':
        lines.append(f'ExposureTime = {exptime}\n')
    lines.append('DateTime = 13-Feb-2026  22:45:30\n')  # padding, matches real mdocs
    return lines


def _build_mdoc(n=10, exptime=None, order='normal'):
    lines = []
    for i in range(n):
        lines += _mdoc_section(i, i * 3.0, 4.16, f'f{i}', exptime=exptime, order=order)
    return lines


def test_conformance_no_exptime_field_at_all():
    """Old-style mdoc (no ExposureTime anywhere): passes <2.3.0, fails >=2.3.0."""
    result = check_parser_conformance(_build_mdoc(exptime=None))
    assert result['passes_v222'] is True
    assert result['passes_v230'] is False
    assert result['n_tilts_v222'] == 10
    assert result['n_tilts_v230'] == 0


def test_conformance_exptime_correct_order():
    """ExposureTime present, correctly after SubFramePath: passes both."""
    result = check_parser_conformance(_build_mdoc(exptime=1.0, order='normal'))
    assert result['passes_v222'] is True
    assert result['passes_v230'] is True
    assert result['n_tilts_v222'] == 10
    assert result['n_tilts_v230'] == 10


def test_conformance_exptime_wrong_order():
    """
    ExposureTime present but before SubFramePath: passes <2.3.0 (which never
    looks for it), fails >=2.3.0 with sections silently merging (the
    ts-069/ts-001 real-world failure signature -- roughly half the tilts
    recovered, not zero and not all).
    """
    result = check_parser_conformance(_build_mdoc(exptime=1.0, order='before_path'))
    assert result['passes_v222'] is True
    assert result['passes_v230'] is False
    assert result['n_tilts_v222'] == 10
    assert 0 < result['n_tilts_v230'] < 10
