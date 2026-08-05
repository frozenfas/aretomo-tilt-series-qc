"""
Tests for shared/discovery.py:parse_fraction_filename() -- consolidates
check_gain_transform.py's _FNAME_RE and validate_mdoc.py's _SFNAME_RE,
which used to accept different strings (one required exactly '.tiff'
with a looser tilt-angle token; the other accepted '.tif'/'.tiff' with a
strict decimal tilt token). A filename one accepted, the other could
silently reject or mis-handle.

Synthetic fixtures (no mounted data required) so these always run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aretomo3_preprocess.shared.discovery import parse_fraction_filename
from aretomo3_preprocess.commands.check_gain_transform import _parse_movie_name
from aretomo3_preprocess.commands.validate_mdoc import _parse_subframes_file


def test_basic_tiff():
    acq, tilt = parse_fraction_filename('Position_1_001_14.00_20260213_171849_fractions.tiff')
    assert (acq, tilt) == (1, 14.00)


def test_tif_extension_accepted():
    """Both consolidated regexes' union: check_gain_transform.py already
    accepted .tif; validate_mdoc.py's old _SFNAME_RE required exactly
    .tiff -- now both accept either via the same function."""
    acq, tilt = parse_fraction_filename('Position_1_001_14.00_20260213_171849_fractions.tif')
    assert (acq, tilt) == (1, 14.00)


def test_negative_tilt():
    acq, tilt = parse_fraction_filename('Position_1_003_-6.00_20260213_171849_fractions.tiff')
    assert (acq, tilt) == (3, -6.00)


def test_case_insensitive_extension():
    acq, tilt = parse_fraction_filename('Position_1_001_0.00_20260213_171849_FRACTIONS.TIFF')
    assert (acq, tilt) == (1, 0.00)


def test_full_path_search_finds_match_at_end():
    """check_gain_transform.py calls this on Path.name (a bare filename),
    but the underlying regex uses search(), not an anchored match -- a
    full path-like string with the pattern only at the end still works."""
    acq, tilt = parse_fraction_filename('/some/dir/Position_1_001_14.00_20260213_171849_fractions.tiff')
    assert (acq, tilt) == (1, 14.00)


def test_bare_integer_tilt_rejected():
    """A real SerialEM tilt token always has a decimal point -- a bare
    integer is never actually valid, only ever a malformed/unrelated
    match. This is the STRICT half of the union (check_gain_transform.py's
    original behavior); validate_mdoc.py's old looser [-\\d.]+ token would
    have silently accepted this."""
    acq, tilt = parse_fraction_filename('Position_1_001_14_20260213_171849_fractions.tiff')
    assert (acq, tilt) == (None, None)


def test_no_match_returns_none_none():
    assert parse_fraction_filename('not_a_movie_file.txt') == (None, None)


def test_eer_extension_not_matched():
    """Neither original regex matched .eer -- confirming this stays
    unchanged (not silently broadening scope beyond what was consolidated)."""
    assert parse_fraction_filename(
        'Position_1_001_14.00_20260213_171849_fractions.eer') == (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Both real call sites now agree on the same filename
# ─────────────────────────────────────────────────────────────────────────────

def test_check_gain_transform_and_validate_mdoc_agree_on_tif(tmp_path):
    """The concrete scenario the audit flagged: a .tif file (not .tiff)
    that check_gain_transform.py already accepted but validate_mdoc.py's
    --fix-subframes rebuild used to silently reject."""
    fname = 'Position_1_001_14.00_20260213_171849_fractions.tif'

    class _FakePath:
        name = fname
    cg_result = _parse_movie_name(_FakePath())

    txt_path = tmp_path / 'subframes.txt'
    txt_path.write_text(fname + '\n')
    vm_result = _parse_subframes_file(txt_path)

    assert cg_result == (1, 14.00)
    assert vm_result == [(1, 14.00, fname)]
