"""
Tests for the alpha_offset handling convention introduced in v2:
AreTomo3 never bakes AlphaOffset into the TILT column (.aln data rows or
_st.tlt) -- it's always a separate, header-only correction that consumers
apply explicitly. See aln_edit.py, pytom_match.py, gapstop_match.py,
relion5_convert.py, validate_mdoc.py.

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
        lines.append(f'{i}  87.29  1.00000  0.0  0.0  1.00  1.00  1.00  0.00  {tilt:.2f}\n')
    path.write_text(''.join(lines))


def _write_tlt(path: Path):
    lines = []
    for i, tilt in enumerate(_NOMINAL_TILTS, start=1):
        lines.append(f'{tilt:8.2f}   {i}   1.0000\n')
    path.write_text(''.join(lines))


@pytest.fixture(params=[0.0, -14.0], ids=['alpha=0 (v1 parity)', 'alpha=-14 (v2 correction)'])
def synthetic_ts(tmp_path, request):
    """A synthetic ts-001.aln + ts-001_TLT.txt pair with the given alpha_offset."""
    alpha = request.param
    _write_aln(tmp_path / 'ts-001.aln', alpha)
    _write_tlt(tmp_path / 'ts-001_TLT.txt')
    return tmp_path, alpha


# ─────────────────────────────────────────────────────────────────────────────
# pytom_match.py / gapstop_match.py: tilt angles = nominal + alpha_offset
# ─────────────────────────────────────────────────────────────────────────────

def test_pytom_match_applies_alpha_offset(synthetic_ts):
    aretomo_dir, alpha = synthetic_ts
    tlt_out, defocus_out, exposure_out = pytom_match._read_ts_metadata(
        aretomo_dir, 'ts-001')
    expected = [t + alpha for t in _NOMINAL_TILTS]
    assert tlt_out == pytest.approx(expected, abs=1e-6)
    if alpha == 0.0:
        # v1 parity: with no offset, tilt_out must equal the raw nominal
        # values exactly -- this is the behavior before the alpha_offset
        # convention change, unaffected by it.
        assert tlt_out == pytest.approx(_NOMINAL_TILTS, abs=1e-6)


def test_gapstop_match_applies_alpha_offset(synthetic_ts):
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
# aln-edit: only the AlphaOffset header changes; TILT column never does
# ─────────────────────────────────────────────────────────────────────────────

def test_aln_edit_only_touches_header(tmp_path, monkeypatch):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = tmp_path / 'ts-001.aln'
    _write_aln(aln_path, alpha_offset=0.0)
    original_data_lines = [
        l for l in aln_path.read_text().splitlines() if l and not l.startswith('#')
    ]

    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)

    new_text = aln_path.read_text()
    new_data_lines = [l for l in new_text.splitlines() if l and not l.startswith('#')]
    assert new_data_lines == original_data_lines, \
        'aln-edit must not modify TILT column data rows'

    assert '# AlphaOffset =  -14.00' in new_text or '# AlphaOffset =   -14.00' in new_text

    # backup preserves the true original (alpha=0)
    bak_path = aln_path.with_suffix(aln_path.suffix + '.bak')
    assert bak_path.exists()
    assert '# AlphaOffset =     0.00' in bak_path.read_text()


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
