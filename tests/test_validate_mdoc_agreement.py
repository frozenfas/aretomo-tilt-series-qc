"""
Tests for validate-mdoc's cross-check between its own destructive-
simulation section count and shared/parsers.py:parse_mdoc_file's real,
mdocfile-library parse (the parser that actually populates project.json's
mdoc_data). A file that passes the simulation but disagrees with mdocfile
on section count means the data saved to project.json may not reflect
what AreTomo3 actually does with the file -- flagged as a real, if rare,
risk by run_aretomo3.py's own code comments before this check existed.

Synthetic fixtures (no mounted data required) so these always run. The
wiring is tested via monkeypatching check_mdocfile_agreement directly --
exercising the real mdocfile library's parsing quirks is out of scope
here (validated separately against 484 real mdocs, see commit history).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import aretomo3_preprocess.commands.validate_mdoc as vm
from aretomo3_preprocess.commands.validate_mdoc import validate_file


def _section(zval, tilt, fname, dose=4.16, exptime=1.0):
    return (
        f'[ZValue = {zval}]\n'
        f'TiltAngle = {tilt}\n'
        f'ExposureDose = {dose}\n'
        f'SubFramePath = \\\\ACQPC\\staging\\{fname}\n'
        f'ExposureTime = {exptime}\n'
        f'DateTime = 01-Jan-2026  00:00:00\n'
    )


_FILENAMES = [f'Position_1_{i:03d}_0.0_fractions.tiff' for i in range(7)]


def _write_valid_mdoc(path: Path):
    text = ''.join(_section(i, float(i), f) for i, f in enumerate(_FILENAMES))
    path.write_text(text)


def test_agreement_passes_when_counts_match(tmp_path, monkeypatch):
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_valid_mdoc(mdoc_path)
    monkeypatch.setattr(vm, 'check_mdocfile_agreement',
                        lambda path: {'n_mdocfile': 7})

    r = validate_file(str(mdoc_path))
    assert r['success'] is True


def test_agreement_fails_when_counts_disagree(tmp_path, monkeypatch):
    """The scenario the check exists for: simulation says 7 tilts, but the
    mdocfile-based real parser (the one that actually populates
    project.json) sees a different number -- must not silently pass."""
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_valid_mdoc(mdoc_path)
    monkeypatch.setattr(vm, 'check_mdocfile_agreement',
                        lambda path: {'n_mdocfile': 14})

    r = validate_file(str(mdoc_path))
    assert r['success'] is False
    assert any('mdocfile' in issue for issue in r['issues'])


def test_agreement_skipped_when_mdocfile_not_installed(tmp_path, monkeypatch):
    """n_mdocfile=None (mdocfile not installed) must not be treated as a
    disagreement -- it means 'can't check', not 'counts differ'."""
    mdoc_path = tmp_path / 'Position_1.mdoc'
    _write_valid_mdoc(mdoc_path)
    monkeypatch.setattr(vm, 'check_mdocfile_agreement',
                        lambda path: {'n_mdocfile': None})

    r = validate_file(str(mdoc_path))
    assert r['success'] is True


def test_agreement_not_checked_for_already_failing_file(tmp_path, monkeypatch):
    """A file already failing (e.g. too few sections) doesn't need a second
    opinion from mdocfile -- check_mdocfile_agreement should not even be
    called."""
    mdoc_path = tmp_path / 'Position_1.mdoc'
    # Only 3 sections -- fails _MIN_TILTS regardless of mdocfile agreement.
    text = ''.join(_section(i, float(i), f) for i, f in enumerate(_FILENAMES[:3]))
    mdoc_path.write_text(text)

    calls = []
    monkeypatch.setattr(vm, 'check_mdocfile_agreement',
                        lambda path: calls.append(path) or {'n_mdocfile': 3})

    r = validate_file(str(mdoc_path))
    assert r['success'] is False
    assert calls == []


def test_fix_dose_rechecks_agreement_against_post_fix_count(tmp_path, monkeypatch):
    """--fix-order/--fix-dose/--fix-exptime can change the simulation's own
    recovered tilt count (that's the point of --fix-order) -- the
    agreement re-check after a fix must compare against the POST-fix
    count, not stay stuck on a stale pre-fix value."""
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

    # mdocfile always reports 7 (it doesn't care about ExposureDose being
    # missing) -- agreement should hold once the fix recovers 7 tilts too.
    monkeypatch.setattr(vm, 'check_mdocfile_agreement',
                        lambda path: {'n_mdocfile': 7})

    r = validate_file(str(mdoc_path), fix_dose=True, dose=4.16)
    assert r['fixed'] is True
    assert r['n_tilts'] == 7
    assert r['success'] is True
