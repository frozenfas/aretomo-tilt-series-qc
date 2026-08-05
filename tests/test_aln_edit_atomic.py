"""
Tests for aln_edit.py's atomic writes and half-applied-pair detection.

Audit finding: the .aln and its companion IMOD .tlt were updated as two
separate, non-atomic writes. A crash between them left a half-applied
offset (one file shifted, the other not) with nothing to detect that
state on the next run.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.commands import aln_edit


_NOMINAL_TILTS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def _write_aln(path: Path, alpha_offset: float = 0.0):
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


def _write_tlt_plain(path: Path):
    path.write_text(''.join(f'{t:8.2f}\n' for t in _NOMINAL_TILTS))


def _make_ts(tmp_path, ts_name='ts-001', with_imod=True):
    aln_path = tmp_path / f'{ts_name}.aln'
    _write_aln(aln_path, alpha_offset=0.0)
    if with_imod:
        imod_dir = tmp_path / f'{ts_name}_Imod'
        imod_dir.mkdir()
        _write_tlt_plain(imod_dir / f'{ts_name}_st.tlt')
    return aln_path


# ─────────────────────────────────────────────────────────────────────────────
# _atomic_write_text
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_write_text_writes_correct_content(tmp_path):
    p = tmp_path / 'file.txt'
    aln_edit._atomic_write_text(p, 'hello world\n')
    assert p.read_text() == 'hello world\n'


def test_atomic_write_text_leaves_no_tmp_file_behind(tmp_path):
    p = tmp_path / 'file.txt'
    aln_edit._atomic_write_text(p, 'content\n')
    leftovers = list(tmp_path.glob('*.tmp*'))
    assert leftovers == []


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    p = tmp_path / 'file.txt'
    p.write_text('old content\n')
    aln_edit._atomic_write_text(p, 'new content\n')
    assert p.read_text() == 'new content\n'


# ─────────────────────────────────────────────────────────────────────────────
# _find_half_applied_pairs
# ─────────────────────────────────────────────────────────────────────────────

def test_no_half_applied_pairs_on_fresh_files(tmp_path):
    aln_path = _make_ts(tmp_path)
    assert aln_edit._find_half_applied_pairs([aln_path]) == []


def test_no_half_applied_pairs_after_clean_full_run(tmp_path, monkeypatch):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = _make_ts(tmp_path)
    aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)
    assert aln_edit._find_half_applied_pairs([aln_path]) == []


def test_detects_half_applied_pair(tmp_path):
    """Simulates a crash between the .aln write (backed up + rewritten)
    and the paired IMOD file ever being touched -- aln has a .bak, the
    IMOD file exists, but the IMOD .bak was never created."""
    aln_path = _make_ts(tmp_path, with_imod=True)
    aln_bak = aln_path.with_suffix(aln_path.suffix + '.bak')
    aln_bak.write_text(aln_path.read_text())  # simulate: aln was backed up...
    aln_path.write_text(aln_path.read_text().replace('0.00', '-14.00'))  # ...and rewritten
    # ...but the IMOD companion's .bak was never created (crash before that step).

    assert aln_edit._find_half_applied_pairs([aln_path]) == [aln_path]


def test_no_imod_companion_is_not_half_applied(tmp_path):
    """A TS with no IMOD _st.tlt at all is never 'half applied' -- there's
    no pair to be inconsistent."""
    aln_path = _make_ts(tmp_path, with_imod=False)
    aln_bak = aln_path.with_suffix(aln_path.suffix + '.bak')
    aln_bak.write_text(aln_path.read_text())
    assert aln_edit._find_half_applied_pairs([aln_path]) == []


def test_apply_offset_refuses_when_half_applied_detected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(aln_edit, '_confirm', lambda prompt: True)
    aln_path = _make_ts(tmp_path, with_imod=True)
    aln_bak = aln_path.with_suffix(aln_path.suffix + '.bak')
    aln_bak.write_text(aln_path.read_text())
    aln_path.write_text(aln_path.read_text().replace('0.00', '-14.00'))

    with pytest.raises(SystemExit) as exc_info:
        aln_edit._apply_offset([aln_path], offset=-14.0, dry_run=False)
    assert exc_info.value.code == 1

    out = capsys.readouterr().out
    assert 'half-applied' in out.lower()
    assert aln_path.name in out
