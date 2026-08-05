"""
validate-mdoc — check and optionally repair SerialEM mdoc files for AreTomo3.

What it checks
--------------
As of AreTomo3 2.3.0, CReadMdoc::DoIt reads FIVE fields per tilt section
using a strict, sequential, destructive state machine:

    ZValue  →  TiltAngle  →  ExposureDose  →  SubFramePath  →  ExposureTime

("Destructive" meaning each field is searched for only after the previous
one is found, and lines are consumed as they're read -- the parser never
backtracks.) If *any* of these are absent, OR if they appear in a different
relative order than listed above within a section, the parser stalls: it
keeps consuming lines -- including the next section's [ZValue] header and
its own fields -- until it happens to find a line matching whatever field
it's still waiting for. This silently merges two (or more) tilt sections
into a single output entry instead of failing outright, which is worse than
an outright failure because the file "succeeds" with corrupted acquisition
indices.

ExposureTime was added to this validator after a real-world case (project
bi38262-21-akinetes, ts-069) where every section had ExposureTime present
(29 sections, 29 ExposureTime occurrences -- a simple presence/count check
passes cleanly) but AreTomo3 2.3.0 still only recovered 14 of 29 tilts, with
acquisition indices stepping by 2 each time. The field was present but out
of the order this parser hard-codes, most likely with SubFramePath appearing
before ExposureDose in that project's mdoc convention -- exactly the
"present but wrong order" case count-only checks can't catch. Versions of
AreTomo3 before 2.3.0 only require the first four fields (through
SubFramePath) and know nothing about ExposureTime, so mdocs written for
older pipelines are especially likely to have it absent or out of place.

The most common failure mode is a missing ExposureDose field.  This arises
when mdoc files are exported or reconstructed by software that does not inject
dose metadata (SerialEM only writes ExposureDose when dose calibration is
active).

ExposureDose definition (SerialEM / IMOD documentation)
--------------------------------------------------------
ExposureDose is the **total incident electron dose for the entire tilt
exposure** (all sub-frames combined), in units of e⁻/Å².  It is NOT the
per-frame dose.  The per-frame dose is stored separately in the
FrameDosesAndNumber field.

Example cross-check: ExposureDose = 0.52 with NumSubFrames = 8 →
per-frame dose ≈ 0.065 e⁻/Å², consistent with low-dose K3 tilt series.

Usage
-----
  # Validate only — report issues (missing dose, missing exptime, wrong
  # order) and the exact fix command to run for each, without changing
  # anything:
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc frames/ts-129.mdoc
  aretomo3-preprocess validate-mdoc frames/ts-*.mdoc

  # Fields present but in the wrong order (no value needed):
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc --fix-order

  # ExposureDose missing entirely:
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc --fix-dose --dose 0.52

  # ExposureTime missing entirely (AreTomo3 >= 2.3.0 only):
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc --fix-exptime --exptime 1.0

  # Any combination, applied together in one pass (order first, then
  # missing fields are injected into the now-reordered file):
  aretomo3-preprocess validate-mdoc frames/ts-*.mdoc --fix-order --fix-dose --dose 0.52 --fix-exptime --exptime 1.0

Backup
------
Before any modification the original file is copied to
  <filename>.mdoc.original.bak
A second run will not overwrite an existing .bak (safety).
"""

import re
import sys
import shutil
import argparse
from pathlib import Path

from aretomo3_preprocess.shared.discovery import parse_fraction_filename


# ─────────────────────────────────────────────────────────────────────────────
# AreTomo3 parser constants (from CReadMdoc.cpp)
# ─────────────────────────────────────────────────────────────────────────────

_LINE_BUF = 256   # char acBuf[256] in C++ source
_MIN_TILTS = 7    # if(m_iNumTilts >= 7) return true


# ─────────────────────────────────────────────────────────────────────────────
# Low-level field extractors — mirror the C++ logic exactly
# ─────────────────────────────────────────────────────────────────────────────

def _atof(s):
    m = re.match(r'[+-]?\d+(\.\d*)?([eE][+-]?\d+)?', s.lstrip())
    return float(m.group()) if m else 0.0


def _extract_val_z(line):
    if 'ZValue' not in line:
        return -99
    idx = line.rfind('=')
    if idx < 0:
        return -99
    rest = line[idx + 1:].lstrip()
    m = re.match(r'-?\d+', rest)
    return int(m.group()) if m else -99


def _extract_tilt(line):
    if 'TiltAngle' not in line:
        return False, 0.0
    idx = line.rfind('=')
    if idx < 0:
        return False, 0.0
    return True, _atof(line[idx + 1:])


def _extract_dose(line):
    if 'ExposureDose' not in line:
        return False, 0.0
    idx = line.rfind('=')
    if idx < 0:
        return False, 0.0
    if len(line[idx:]) < 2:
        return False, 0.0
    return True, _atof(line[idx + 1:])


def _extract_frame_path(line):
    if 'SubFramePath' not in line:
        return None
    idx = line.rfind('=')
    if idx < 0:
        return None
    path = line[idx + 1:].rstrip('\n\r').rstrip()
    return path.strip() or None


def _extract_exptime(line):
    """Mirrors CReadMdoc::mExtractExpTime (AreTomo3 >= 2.3.0 only)."""
    if 'ExposureTime' not in line:
        return False, 0.0
    idx = line.rfind('=')
    if idx < 0:
        return False, 0.0
    if len(line[idx:]) < 2:
        return False, 0.0
    return True, _atof(line[idx + 1:])


def _frame_basename(raw_path: str) -> str:
    """
    SubFramePath is whatever the acquisition PC wrote -- typically a
    Windows/UNC path (\\\\server\\...\\file.eer) that no longer resolves on
    the processing machine. AreTomo3 only ever looks for the referenced
    movie alongside the mdoc itself (rename-ts's own symlinks are created
    "alongside the originals" for exactly this reason), so only the
    filename component is useful here.
    """
    return re.split(r'[\\/]+', raw_path.strip())[-1]


def check_mdocfile_agreement(mdoc_path) -> dict:
    """
    Cross-check this file's own destructive-simulation section count
    against shared/parsers.py:parse_mdoc_file's real, mdocfile-library
    parse -- the SAME function that actually populates mdoc_data in
    project.json (via _save_mdoc_to_project below). These are two
    independent implementations with genuinely different purposes
    (_simulate_aretomo3 faithfully mimics AreTomo3's own destructive C++
    parser; parse_mdoc_file does a robust real parse via the mdocfile
    library), so agreement isn't guaranteed by construction --
    disagreement means the data actually saved to project.json may not
    reflect what AreTomo3 will do with this file, a distinct risk from
    either parser being independently "wrong" (run_aretomo3.py's own code
    comments already flagged this as a known, previously-unchecked risk).

    Empirically verified zero disagreements across 484 real mdocs that
    otherwise pass validate-mdoc's checks before making this blocking
    (2026-08) -- a theoretical risk worth guarding against even though it
    wasn't observed to currently manifest.

    Returns {'n_mdocfile': int|None}. None means mdocfile isn't installed
    (parse_mdoc_file degrades gracefully to an empty dict in that case) --
    treated by the caller as "can't check", not a disagreement.
    """
    from aretomo3_preprocess.shared.parsers import parse_mdoc_file, _HAS_MDOCFILE
    if not _HAS_MDOCFILE:
        return {'n_mdocfile': None}
    frames, _, _ = parse_mdoc_file(mdoc_path)
    return {'n_mdocfile': len(frames)}


def check_frames_found(mdoc_path, lines) -> dict:
    """
    Verify every SubFramePath-referenced movie file actually exists next to
    mdoc_path. AreTomo3 requires raw movies and their mdoc to be co-located,
    so the frames directory is always mdoc_path's own parent -- never
    whatever directory SubFramePath itself encodes.

    Returns {'n_frames': int, 'n_found': int, 'missing': [filename, ...]}.
    """
    frames_dir = Path(mdoc_path).resolve().parent
    missing = []
    n_frames = 0
    for line in lines:
        raw = _extract_frame_path(line)
        if raw is None:
            continue
        n_frames += 1
        fname = _frame_basename(raw)
        if not (frames_dir / fname).exists():
            missing.append(fname)
    return {'n_frames': n_frames, 'n_found': n_frames - len(missing), 'missing': missing}


# ─────────────────────────────────────────────────────────────────────────────
# Simulate CReadMdoc::DoIt
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_aretomo3(lines, require_exptime=True):
    """
    Simulate the AreTomo3 mdoc parser state machine.

    require_exptime=True  (default) simulates AreTomo3 >= 2.3.0, which reads
    a 5th field (ExposureTime) after SubFramePath.  require_exptime=False
    simulates AreTomo3 < 2.3.0 (e.g. 2.2.2), which only ever reads the first
    four fields and has no notion of ExposureTime at all -- its presence or
    position anywhere in the file is irrelevant to that parser.

    Returns (n_tilts_loaded, failure_state) where failure_state is one of:
      None                    — success
      'no_zvalue'             — no [ZValue = N] sections found
      'missing_tiltangle'     — stuck waiting for TiltAngle
      'missing_exposuredose'  — stuck waiting for ExposureDose
      'missing_subframepath'  — stuck waiting for SubFramePath
      'missing_exposuretime'  — stuck waiting for ExposureTime (only possible
                                 when require_exptime=True)
      'too_few_tilts'         — parsed OK but fewer than 7 tilts

    NOTE: unlike a simple per-field presence count, this simulates the real
    parser's destructive, sequential behavior. A field that is present
    somewhere in the file but out of order (e.g. SubFramePath appearing
    before ExposureDose within a section) will NOT be caught by presence
    counts alone -- it shows up here as sections silently merging (fewer
    tilts than [ZValue] sections, with no explicit failure reported), which
    is exactly the "present but wrong order" case that caused AreTomo3
    2.3.0 to only recover half of ts-069's 29 real tilts despite
    ExposureTime being present in every section.
    """
    n = 0
    li = 0
    total = len(lines)
    last_failure = None

    while li < total:
        line = lines[li]; li += 1
        if _extract_val_z(line) < 0:
            continue

        bTilt = bDose = bFm = False
        bExpTime = not require_exptime   # pre-satisfied when not required
        while li < total:
            line = lines[li]; li += 1
            if not bTilt:
                ok, _ = _extract_tilt(line)
                if ok:
                    bTilt = True
            elif not bDose:
                ok, _ = _extract_dose(line)
                if ok:
                    bDose = True
            elif not bFm:
                if _extract_frame_path(line) is not None:
                    bFm = True
            elif not bExpTime:
                ok, _ = _extract_exptime(line)
                if ok:
                    bExpTime = True
            else:
                n += 1
                break
        else:
            # Inner loop hit EOF
            if not bTilt:
                last_failure = 'missing_tiltangle'
            elif not bDose:
                last_failure = 'missing_exposuredose'
            elif not bFm:
                last_failure = 'missing_subframepath'
            elif not bExpTime:
                last_failure = 'missing_exposuretime'
            elif bExpTime:
                n += 1   # last section at EOF — commit it
            break

    if n >= _MIN_TILTS:
        return n, None
    if last_failure:
        return n, last_failure
    if n == 0:
        return n, 'no_zvalue'
    return n, 'too_few_tilts'


# ─────────────────────────────────────────────────────────────────────────────
# Structural field scan (independent of state machine)
# ─────────────────────────────────────────────────────────────────────────────

def _scan_fields(lines):
    """
    Count key fields to give a clear picture of what is present/absent.
    Returns a dict with counts and a list of human-readable issue strings.
    """
    n_zval   = sum(1 for l in lines if '[ZValue' in l)
    n_tilt   = sum(1 for l in lines if 'TiltAngle'    in l and '=' in l and '[' not in l)
    n_dose   = sum(1 for l in lines if 'ExposureDose' in l and '=' in l)
    n_path   = sum(1 for l in lines if 'SubFramePath' in l and '=' in l)
    n_exptime = sum(1 for l in lines if 'ExposureTime' in l and '=' in l)

    # Which sections are missing ExposureDose?
    missing_dose_sections = []
    current_z = None
    has_dose_in_section = False
    for line in lines:
        z = _extract_val_z(line)
        if z >= 0:
            if current_z is not None and not has_dose_in_section:
                missing_dose_sections.append(current_z)
            current_z = z
            has_dose_in_section = False
        if current_z is not None and 'ExposureDose' in line and '=' in line:
            has_dose_in_section = True
    if current_z is not None and not has_dose_in_section:
        missing_dose_sections.append(current_z)

    # Which sections are missing ExposureTime? (presence-only check; does NOT
    # catch the "present but wrong relative order" case -- that requires the
    # full sequential simulation in _simulate_aretomo3.)
    missing_exptime_sections = []
    current_z = None
    has_exptime_in_section = False
    for line in lines:
        z = _extract_val_z(line)
        if z >= 0:
            if current_z is not None and not has_exptime_in_section:
                missing_exptime_sections.append(current_z)
            current_z = z
            has_exptime_in_section = False
        if current_z is not None and 'ExposureTime' in line and '=' in line:
            has_exptime_in_section = True
    if current_z is not None and not has_exptime_in_section:
        missing_exptime_sections.append(current_z)

    issues = []
    if n_dose == 0 and n_exptime > 0:
        issues.append(
            'ExposureDose absent in all {} section(s); ExposureTime present '
            '({} occurrences) — likely missing dose calibration in SerialEM'.format(
                n_zval, n_exptime)
        )
    elif n_dose == 0:
        issues.append('ExposureDose absent in all {} section(s)'.format(n_zval))
    elif missing_dose_sections:
        issues.append(
            'ExposureDose missing from {} section(s): ZValue {}'.format(
                len(missing_dose_sections),
                missing_dose_sections[:10],
            )
        )

    if n_tilt < n_zval:
        issues.append(
            'TiltAngle present {}× but {} sections'.format(n_tilt, n_zval)
        )
    if n_path < n_zval:
        issues.append(
            'SubFramePath present {}× but {} sections'.format(n_path, n_zval)
        )
    if n_exptime < n_zval:
        issues.append(
            'ExposureTime present {}× but {} sections (required by AreTomo3 '
            '>= 2.3.0)'.format(n_exptime, n_zval)
        )
    elif missing_exptime_sections:
        # Shouldn't normally happen if n_exptime == n_zval, but section-level
        # accounting can differ from a flat count if lines are malformed.
        issues.append(
            'ExposureTime missing from {} section(s): ZValue {}'.format(
                len(missing_exptime_sections),
                missing_exptime_sections[:10],
            )
        )
    if n_zval < _MIN_TILTS:
        issues.append(
            'Only {} ZValue section(s) — minimum is {}'.format(n_zval, _MIN_TILTS)
        )

    long_lines = [l for l in lines
                  if len(l.encode('latin-1', errors='replace')) >= _LINE_BUF - 1]
    if long_lines:
        issues.append(
            '{} line(s) >= {} chars (AreTomo3 buffer limit — '
            'SubFramePath may be truncated)'.format(len(long_lines), _LINE_BUF - 1)
        )

    return dict(
        n_sections=n_zval, n_tilt=n_tilt, n_dose=n_dose, n_path=n_path,
        n_exptime=n_exptime, missing_dose_sections=missing_dose_sections,
        missing_exptime_sections=missing_exptime_sections,
        issues=issues,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Field ORDER check — independent of presence/absence.
#
# A field can be present in every section and still break AreTomo3's parser
# if it's in the wrong relative position, because CReadMdoc::DoIt searches
# for each field only after the previous one, consuming lines as it goes.
# This is checked structurally (by position within each section) rather
# than by re-running the destructive simulation, because a bad order can
# still yield >= _MIN_TILTS merged "tilts" and read as a simulation SUCCESS
# even though the data is corrupted -- exactly what happened with ts-069.
# ─────────────────────────────────────────────────────────────────────────────

_CANONICAL_ORDER = ['TiltAngle', 'ExposureDose', 'SubFramePath', 'ExposureTime']
# AreTomo3 < 2.3.0 never reads ExposureTime, so its position is irrelevant
# to that parser -- only the first three fields' relative order matters.
_CANONICAL_ORDER_V222 = ['TiltAngle', 'ExposureDose', 'SubFramePath']

_FIELD_CHECKERS = {
    'TiltAngle':     lambda l: _extract_tilt(l)[0],
    'ExposureDose':  lambda l: _extract_dose(l)[0],
    'SubFramePath':  lambda l: _extract_frame_path(l) is not None,
    'ExposureTime':  lambda l: _extract_exptime(l)[0],
}


def _split_sections(lines):
    """
    Split lines into (header_line, body_lines) per [ZValue] section.
    Returns a list of (zval, header_line, body_lines) tuples. Lines before
    the first [ZValue] are not included (there is no section to attach them
    to).
    """
    sections = []
    i = 0
    total = len(lines)
    while i < total:
        z = _extract_val_z(lines[i])
        if z < 0:
            i += 1
            continue
        header = lines[i]
        i += 1
        body = []
        while i < total and _extract_val_z(lines[i]) < 0:
            body.append(lines[i])
            i += 1
        sections.append((z, header, body))
    return sections


def _field_positions(body_lines, fields=None):
    """
    Return an ordered dict {field_name: line} for the FIRST occurrence of
    each field in `fields` (default: all four) found in body_lines, in the
    order they actually appear (encounter order, not canonical order).
    """
    checkers = {k: v for k, v in _FIELD_CHECKERS.items()
                if fields is None or k in fields}
    found = {}
    for line in body_lines:
        for name, check in checkers.items():
            if name in found:
                continue
            if check(line):
                found[name] = line
                break
    return found


def _check_order(lines, canonical_order=_CANONICAL_ORDER):
    """
    Check every section's field order against `canonical_order`.

    Pass canonical_order=_CANONICAL_ORDER_V222 to check against AreTomo3
    < 2.3.0's rules instead (ExposureTime ignored entirely).

    Returns a list of dicts, one per out-of-order section:
      {zval, found_order, expected_order}
    Only fields that are actually PRESENT in a section are compared --
    missing fields are a separate problem (see _scan_fields) and are
    skipped here rather than counted as "out of order".
    """
    out_of_order = []
    for zval, header, body in _split_sections(lines):
        found = _field_positions(body, fields=canonical_order)
        if len(found) < 2:
            continue  # nothing meaningful to compare
        encounter_order = list(found.keys())
        expected_order = [f for f in canonical_order if f in found]
        if encounter_order != expected_order:
            out_of_order.append(dict(
                zval=zval,
                found_order=encounter_order,
                expected_order=expected_order,
            ))
    return out_of_order


def _describe_order_issue(out_of_order):
    """Build a short, human-readable summary of an order-check result."""
    if not out_of_order:
        return None
    example = out_of_order[0]
    return (
        '{} section(s) have fields out of order (e.g. ZValue {}: found {}, '
        'AreTomo3 expects {}) — parser will silently merge sections instead '
        'of failing outright'.format(
            len(out_of_order), example['zval'],
            ' → '.join(example['found_order']),
            ' → '.join(example['expected_order']),
        )
    )


def check_parser_conformance(lines):
    """
    Check whether a mdoc would parse correctly under AreTomo3 < 2.3.0
    (4-field, no ExposureTime) versus >= 2.3.0 (5-field, ExposureTime
    required and order-checked).

    A file can pass one and fail the other -- that combination (passes
    2.2.2, fails 2.3.0) is exactly the "worked fine for years, then broke
    on upgrade" failure mode this module exists to catch pre-emptively.

    Returns a dict:
      passes_v222, passes_v230  — bool
      n_tilts_v222, n_tilts_v230 — int
    """
    n222, fail222 = _simulate_aretomo3(lines, require_exptime=False)
    order222 = _check_order(lines, canonical_order=_CANONICAL_ORDER_V222)
    n230, fail230 = _simulate_aretomo3(lines, require_exptime=True)
    order230 = _check_order(lines, canonical_order=_CANONICAL_ORDER)

    return dict(
        passes_v222=(fail222 is None and not order222),
        passes_v230=(fail230 is None and not order230),
        n_tilts_v222=n222,
        n_tilts_v230=n230,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fix: reorder existing fields into canonical order (no new values needed)
# ─────────────────────────────────────────────────────────────────────────────

def _reorder_section_body(body_lines):
    """
    Reorder the (up to four) required fields within one section's body into
    _CANONICAL_ORDER, without touching any other line's position. The
    reordered fields are consolidated at the position of whichever one
    appeared FIRST in the original body.

    Returns (new_body_lines, changed_bool).
    """
    found = {}
    other = []
    first_target_pos = None

    for line in body_lines:
        matched = None
        for name, check in _FIELD_CHECKERS.items():
            if name in found:
                continue
            if check(line):
                matched = name
                break
        if matched:
            found[matched] = line
            if first_target_pos is None:
                first_target_pos = len(other)
        else:
            other.append(line)

    if len(found) < 2:
        return body_lines, False  # nothing meaningful to reorder

    encounter_order = list(found.keys())
    expected_order = [f for f in _CANONICAL_ORDER if f in found]
    if encounter_order == expected_order:
        return body_lines, False  # already correct

    ordered_targets = [found[name] for name in expected_order]
    new_body = other[:first_target_pos] + ordered_targets + other[first_target_pos:]
    return new_body, True


def _reorder_fields(lines):
    """
    Return a new list of lines with every section's TiltAngle/ExposureDose/
    SubFramePath/ExposureTime fields reordered to match _CANONICAL_ORDER.
    Only sections that are actually out of order are modified; everything
    else (other metadata lines, their relative order, blank lines) is left
    untouched. Sections missing two or more of the four fields are left
    alone (nothing meaningful to reorder -- see _inject_dose/_inject_exptime
    for filling in missing fields first).

    Returns (new_lines, n_reordered).
    """
    out = []
    n_reordered = 0
    i = 0
    total = len(lines)
    while i < total:
        line = lines[i]
        if _extract_val_z(line) < 0:
            out.append(line)
            i += 1
            continue
        out.append(line)  # the [ZValue] header itself
        i += 1
        body = []
        while i < total and _extract_val_z(lines[i]) < 0:
            body.append(lines[i])
            i += 1
        new_body, changed = _reorder_section_body(body)
        if changed:
            n_reordered += 1
        out.extend(new_body)
    return out, n_reordered


# ─────────────────────────────────────────────────────────────────────────────
# Fix: inject ExposureDose into sections where it is absent
# ─────────────────────────────────────────────────────────────────────────────

def _inject_dose(lines, dose_value):
    """
    Return a new list of lines with ExposureDose = <dose_value> inserted
    immediately after each TiltAngle line in any section that lacks it.
    Only modifies sections where ExposureDose is absent; leaves existing
    ExposureDose values untouched.

    The insertion point (after TiltAngle) guarantees AreTomo3's sequential
    parser will find it in the correct order regardless of other field ordering.
    """
    # First build a set of ZValues that are missing ExposureDose
    scan = _scan_fields(lines)
    missing = set(scan['missing_dose_sections'])

    if not missing:
        return lines, 0  # nothing to do

    dose_line = 'ExposureDose = {}\n'.format(dose_value)

    out = []
    n_injected = 0
    current_z = None
    tilt_seen = False
    dose_seen = False

    for line in lines:
        z = _extract_val_z(line)
        if z >= 0:
            current_z = z
            tilt_seen = False
            dose_seen = False

        # Track whether this section already has ExposureDose
        if current_z is not None and 'ExposureDose' in line and '=' in line:
            dose_seen = True

        out.append(line)

        # After TiltAngle in a missing section, inject the dose line
        if (current_z in missing and not tilt_seen and not dose_seen):
            ok, _ = _extract_tilt(line)
            if ok:
                tilt_seen = True
                out.append(dose_line)
                n_injected += 1

    return out, n_injected


# ─────────────────────────────────────────────────────────────────────────────
# Fix: inject ExposureTime into sections where it is absent
# ─────────────────────────────────────────────────────────────────────────────

def _inject_exptime(lines, exptime_value):
    """
    Return a new list of lines with ExposureTime = <exptime_value> inserted
    immediately after each section's SubFramePath line, in any section that
    lacks ExposureTime entirely. Only modifies sections missing it; leaves
    existing ExposureTime values untouched.

    The insertion point (right after SubFramePath) matches the canonical
    order AreTomo3 >= 2.3.0 requires, so this also can't introduce a new
    order problem. Sections that already HAVE ExposureTime but in the wrong
    position are a --fix-order problem, not a --fix-exptime one -- this
    function only fills in a truly absent field.

    A uniform value across every section is safe for AreTomo3's own use of
    this field: it normalizes ExposureTime into a fraction of the series
    total (CReadMdoc::mMakeExpTimeRelative), so identical values everywhere
    just yield equal per-tilt weighting -- the same fallback AreTomo3 itself
    uses when total exposure time is zero.
    """
    scan = _scan_fields(lines)
    missing = set(scan['missing_exptime_sections'])

    if not missing:
        return lines, 0  # nothing to do

    exptime_line = 'ExposureTime = {}\n'.format(exptime_value)

    out = []
    n_injected = 0
    current_z = None
    path_seen = False
    exptime_seen = False

    for line in lines:
        z = _extract_val_z(line)
        if z >= 0:
            current_z = z
            path_seen = False
            exptime_seen = False

        if current_z is not None and 'ExposureTime' in line and '=' in line:
            exptime_seen = True

        out.append(line)

        if (current_z in missing and not path_seen and not exptime_seen):
            if _extract_frame_path(line) is not None:
                path_seen = True
                out.append(exptime_line)
                n_injected += 1

    return out, n_injected


# ─────────────────────────────────────────────────────────────────────────────
# Fix: rebuild mdoc from a SubFramePaths file (too-short / restarted series)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_subframes_file(path):
    """
    Read a SubFramePaths .txt file (one filename per line).
    Returns a list of (acq_order, tilt_angle, filename) sorted by acq_order.
    Filename format: <PositionName>_NNN_TILT_DATE_TIME_fractions.tif[f]

    Uses shared/discovery.py:parse_fraction_filename() for the acq_order/
    tilt extraction -- previously a second, independent regex here
    (_SFNAME_RE) accepted a different set of strings than
    check_gain_transform.py's own copy (this one required exactly
    '.tiff' with a looser tilt-angle token; see CLAUDE.md). Each line is
    just the filename itself (one per line, per the format above), so
    the stripped line doubles as both the string to parse and the
    filename to record once it matches.
    """
    entries = []
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        fname = line.strip()
        acq_order, tilt_angle = parse_fraction_filename(fname)
        if acq_order is not None:
            entries.append((acq_order, tilt_angle, fname))
    entries.sort(key=lambda x: x[0])
    return entries


def _extract_template_section(lines):
    """Return lines of the first ZValue block (excluding the [ZValue = N] header)."""
    in_section = False
    template = []
    for line in lines:
        if _extract_val_z(line) >= 0:
            if in_section:
                break
            in_section = True
            continue
        if in_section:
            template.append(line)
    return template


def _extract_subframe_prefix(lines):
    """Extract the Windows/UNC path prefix from the first SubFramePath field."""
    for line in lines:
        p = _extract_frame_path(line)
        if p:
            idx = max(p.rfind('\\'), p.rfind('/'))
            return p[:idx + 1] if idx >= 0 else ''
    return ''


def _build_section(template_lines, zval, tilt, full_subframe_path, dose):
    """
    Build one mdoc section by cloning the template and substituting
    ZValue, TiltAngle, ExposureDose, and SubFramePath.
    ExposureDose is always written immediately after TiltAngle.
    """
    out = ['[ZValue = {}]\n'.format(zval)]
    dose_written = False
    for line in template_lines:
        key = line.split('=')[0].strip()
        if key == 'TiltAngle':
            out.append('TiltAngle = {:.2f}\n'.format(tilt))
            out.append('ExposureDose = {}\n'.format(dose))
            dose_written = True
        elif key == 'ExposureDose':
            pass  # already written after TiltAngle
        elif key == 'SubFramePath':
            out.append('SubFramePath = {}\n'.format(full_subframe_path))
        else:
            out.append(line)
    if not dose_written:
        out.insert(1, 'ExposureDose = {}\n'.format(dose))
    return out


def _rebuild_from_subframes(lines, subframes_path, dose):
    """
    Rebuild an mdoc using a SubFramePaths .txt file.

    Extracts the header and first section as a template for all per-section
    metadata (StagePosition, Defocus, etc.).  ZValue, TiltAngle, ExposureDose,
    and SubFramePath are set from the SubFramePaths file.

    Returns (new_lines, n_sections, warnings).
    """
    entries = _parse_subframes_file(subframes_path)
    if not entries:
        raise ValueError('No valid filenames found in {}'.format(subframes_path))

    # Sanity check: acq_order in filename should equal ZValue + 1
    warnings = []
    for zval, (acq, tilt, fname) in enumerate(entries):
        if acq != zval + 1:
            warnings.append(
                'Sanity check: {} has acq={}, expected {} (ZValue {}+1)'.format(
                    fname, acq, zval + 1, zval)
            )

    template = _extract_template_section(lines)
    if not template:
        raise ValueError('Could not extract template section from existing mdoc')

    prefix = _extract_subframe_prefix(lines)

    # Collect header lines (everything before the first [ZValue])
    header = []
    for line in lines:
        if _extract_val_z(line) >= 0:
            break
        header.append(line)

    new_lines = list(header)
    for zval, (acq, tilt, fname) in enumerate(entries):
        new_lines.extend(_build_section(template, zval, tilt, prefix + fname, dose))

    return new_lines, len(entries), warnings


# ─────────────────────────────────────────────────────────────────────────────
# Validate (and optionally fix) a single file
# ─────────────────────────────────────────────────────────────────────────────

def validate_file(path, fix_dose=False, fix_exptime=False, fix_order=False,
                  fix_subframes=None, dose=4.16, exptime=1.0,
                  check_frames=False):
    """
    Validate one mdoc file.  Optionally apply fixes.

    fix_dose      — inject missing ExposureDose (--fix-dose, needs --dose)
    fix_exptime   — inject missing ExposureTime (--fix-exptime, needs --exptime)
    fix_order     — reorder existing-but-misplaced fields into canonical
                    order (--fix-order; no value needed, nothing is added
                    or removed, only reordered)
    fix_subframes — path to a directory of SubFramePaths .txt files;
                    used to rebuild mdocs that are too short (--fix-subframes)
    check_frames  — verify every SubFramePath-referenced movie actually
                    exists next to this mdoc (--check-frames). Not fixable
                    by any --fix-* flag, so a missing frame always keeps
                    success False regardless of which other fixes succeed.

    Order is checked independently of presence/absence: a field can be
    present in every section and AreTomo3 will still silently corrupt the
    file if that field is in the wrong relative position (see
    _check_order's docstring). This is reported as an issue even when the
    file "passes" the destructive parser simulation, since a bad order can
    still produce >= _MIN_TILTS merged (wrong) tilts and look like success.

    Fixes are applied in order (reorder → inject dose → inject exptime) so
    that, for example, a file needing both --fix-order and --fix-exptime
    gets its existing fields straightened out before a new one is added
    after SubFramePath. All fixes are combined into a single write with one
    backup, not one write per fix.

    Returns a dict with keys:
      path, success, n_tilts, failure, issues, order_issues, fixed,
      fix_types, n_injected, backed_up, n_sections,
      passes_v222, passes_v230, n_tilts_v222, n_tilts_v230
      (success/n_tilts/failure always reflect the current, >= 2.3.0 rules;
      passes_v222/passes_v230 additionally report conformance to each
      AreTomo3 generation separately -- see check_parser_conformance)
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        return dict(path=path, success=False, n_tilts=0,
                    failure=str(e), issues=[], order_issues=[], fixed=False,
                    fix_types=[], n_injected=0, backed_up=False, n_sections=0,
                    passes_v222=False, passes_v230=False,
                    n_tilts_v222=0, n_tilts_v230=0)

    text = raw.decode('latin-1', errors='replace')
    lines = [l + '\n' for l in text.splitlines()]

    n_tilts, failure = _simulate_aretomo3(lines)
    scan = _scan_fields(lines)
    order_issues = _check_order(lines)
    order_msg = _describe_order_issue(order_issues)
    conformance = check_parser_conformance(lines)

    issues = list(scan['issues'])
    if order_msg:
        issues.append(order_msg)

    frame_check = check_frames_found(path, lines) if check_frames else None
    frames_ok = True
    if frame_check and frame_check['missing']:
        frames_ok = False
        preview = ', '.join(frame_check['missing'][:5])
        if len(frame_check['missing']) > 5:
            preview += ', ...'
        issues.append(
            '{} of {} referenced movie file(s) not found in {}: {}'.format(
                len(frame_check['missing']), frame_check['n_frames'],
                Path(path).resolve().parent, preview))

    # Only worth cross-checking against the mdocfile-based parser once the
    # simulation+order checks already look clean -- a file already known
    # broken doesn't need a second opinion.
    mdocfile_ok = True
    if failure is None and not order_issues:
        agreement = check_mdocfile_agreement(path)
        n_mdocfile = agreement['n_mdocfile']
        if n_mdocfile is not None and n_mdocfile != n_tilts:
            mdocfile_ok = False
            issues.append(
                'Passes AreTomo3-parser simulation ({} tilts) but the '
                'mdocfile library (the parser that actually populates '
                'project.json mdoc_data) parses {} sections -- the data '
                'saved to project.json may not match what AreTomo3 '
                'actually does with this file.'.format(n_tilts, n_mdocfile))

    result = dict(
        path=path,
        # A file that "passes" the simulation but has an order problem is
        # NOT actually fine -- it's exactly the case that produces
        # corrupted-but-successful output. Report it as failing. Same for
        # a movie file that can't be found -- AreTomo3 will fail on it
        # regardless of how clean the mdoc's own fields are. Same for a
        # section-count disagreement with the mdocfile-based parser that
        # actually populates project.json.
        success=(failure is None and not order_issues and frames_ok and mdocfile_ok),
        n_tilts=n_tilts,
        failure=failure,
        issues=issues,
        order_issues=order_issues,
        missing_dose_sections=scan['missing_dose_sections'],
        missing_exptime_sections=scan['missing_exptime_sections'],
        n_sections=scan['n_sections'],
        fixed=False,
        fix_types=[],
        n_injected=0,
        backed_up=False,
        n_frames_expected=frame_check['n_frames'] if frame_check else None,
        n_frames_found=frame_check['n_found'] if frame_check else None,
        **conformance,
    )

    if result['success']:
        return result  # already valid, in order, nothing to do

    # ── Fix subframes (too-short / restarted series) ─────────────────────────
    # Mutually exclusive with the other fixes: a full rebuild, not a patch.
    if fix_subframes is not None and scan['n_sections'] < _MIN_TILTS:
        sf_arg = Path(fix_subframes)
        if sf_arg.is_file():
            sf_path = sf_arg
        else:
            sf_path = sf_arg / '{}.txt'.format(Path(path).stem)
        if not sf_path.exists():
            result['issues'].append(
                '--fix-subframes: no SubFramePaths file at {}'.format(sf_path)
            )
            return result

        bak_path = Path(str(path) + '.original.bak')
        if not bak_path.exists():
            shutil.copy2(path, bak_path)
            result['backed_up'] = True

        try:
            new_lines, n_sections, warnings = _rebuild_from_subframes(
                lines, sf_path, dose)
        except ValueError as e:
            result['issues'].append('Subframe rebuild failed: {}'.format(e))
            return result

        result['issues'].extend(warnings)

        n_after, failure_after = _simulate_aretomo3(new_lines)
        order_after = _check_order(new_lines)
        if failure_after is not None or order_after:
            result['issues'].append(
                'Rebuilt mdoc still fails AreTomo3 simulation ({}) '
                '— file NOT written'.format(failure_after or 'order mismatch')
            )
            return result

        Path(path).write_text(''.join(new_lines), encoding='latin-1')
        result['fixed'] = True
        result['fix_types'] = ['subframes']
        result['n_injected'] = n_sections
        # --fix-subframes rewrites SubFramePath entirely from the .txt file,
        # so re-check against the NEW filenames, not the stale frames_ok
        # computed against the original (about-to-be-discarded) ones.
        if check_frames:
            frame_check_after = check_frames_found(path, new_lines)
            result['success'] = not frame_check_after['missing']
            result['n_frames_expected'] = frame_check_after['n_frames']
            result['n_frames_found'] = frame_check_after['n_found']
            if frame_check_after['missing']:
                result['issues'].append(
                    '{} of {} rebuilt movie file(s) still not found: {}'.format(
                        len(frame_check_after['missing']), frame_check_after['n_frames'],
                        ', '.join(frame_check_after['missing'][:5])))
        else:
            result['success'] = True
        result['n_tilts'] = n_after
        result['failure'] = None
        result['order_issues'] = []
        return result

    # ── Combined pipeline: reorder → inject dose → inject exptime ───────────
    if not (fix_order or fix_dose or fix_exptime):
        return result  # validate-only run, no fix flags given

    if fix_dose and failure not in (None, 'missing_exposuredose') and not order_issues:
        result['issues'].append(
            'Cannot auto-fix failure type "{}" with --fix-dose alone; '
            'check whether --fix-order or --fix-exptime is also needed'.format(failure)
        )
    if fix_exptime and failure not in (None, 'missing_exposuretime') and not order_issues:
        result['issues'].append(
            'Cannot auto-fix failure type "{}" with --fix-exptime alone; '
            'check whether --fix-order or --fix-dose is also needed'.format(failure)
        )

    working = lines
    fix_types = []
    n_injected_total = 0

    if fix_order and order_issues:
        working, n_reordered = _reorder_fields(working)
        if n_reordered:
            fix_types.append('order')
            n_injected_total += n_reordered

    if fix_dose:
        working, n_dose_injected = _inject_dose(working, dose)
        if n_dose_injected:
            fix_types.append('dose')
            n_injected_total += n_dose_injected

    if fix_exptime:
        working, n_exptime_injected = _inject_exptime(working, exptime)
        if n_exptime_injected:
            fix_types.append('exptime')
            n_injected_total += n_exptime_injected

    if not fix_types:
        result['issues'].append(
            'No applicable fix flags matched this file\'s problem(s) — '
            'nothing changed'
        )
        return result

    n_after, failure_after = _simulate_aretomo3(working)
    order_after = _check_order(working)
    if failure_after is not None or order_after:
        result['issues'].append(
            'Fix did not fully resolve the problem (failure={}, '
            '{} section(s) still out of order) — file NOT written'.format(
                failure_after, len(order_after))
        )
        return result

    bak_path = Path(str(path) + '.original.bak')
    if not bak_path.exists():
        shutil.copy2(path, bak_path)
        result['backed_up'] = True
    else:
        result['issues'].append(
            'Backup already exists at {}; skipping backup'.format(bak_path)
        )

    Path(path).write_text(''.join(working), encoding='latin-1')
    result['fixed'] = True
    result['fix_types'] = fix_types
    result['n_injected'] = n_injected_total
    # reorder/dose/exptime never touch SubFramePath, so frames_ok (checked
    # against the original lines above) still applies unchanged. The
    # mdocfile-agreement check does need re-checking against n_after
    # (the fix commonly changes the simulation's own recovered tilt
    # count -- that's the point of --fix-order) even though SubFramePath
    # itself is untouched so the mdocfile-parsed count won't have moved.
    agreement_after = check_mdocfile_agreement(path)
    n_mdocfile_after = agreement_after['n_mdocfile']
    mdocfile_ok_after = n_mdocfile_after is None or n_mdocfile_after == n_after
    if not mdocfile_ok_after:
        result['issues'].append(
            'Fix applied, but mdocfile library still parses {} sections vs '
            '{} tilts recovered by the simulation.'.format(n_mdocfile_after, n_after))
    result['success'] = frames_ok and mdocfile_ok_after
    result['n_tilts'] = n_after
    result['failure'] = None
    result['order_issues'] = []

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'validate-mdoc',
        help='Validate SerialEM mdoc files for AreTomo3 compatibility; '
             'optionally reorder fields and/or inject missing '
             'ExposureDose/ExposureTime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        'mdoc_files', nargs='+', metavar='MDOC',
        help='One or more .mdoc files to validate (glob expansion handled by shell)',
    )
    p.add_argument(
        '--fix-order', action='store_true', default=False,
        help='Reorder existing TiltAngle/ExposureDose/SubFramePath/'
             'ExposureTime fields into the order AreTomo3 >= 2.3.0 expects. '
             'Only reorders fields that are already present -- does not add '
             'or remove anything, and does not need a value. Combine with '
             '--fix-dose/--fix-exptime if fields are also missing outright.',
    )
    p.add_argument(
        '--fix-dose', action='store_true', default=False,
        help='Inject missing ExposureDose into failing files (needs --dose).',
    )
    p.add_argument(
        '--fix-exptime', action='store_true', default=False,
        help='Inject missing ExposureTime into failing files (needs '
             '--exptime). Required by AreTomo3 >= 2.3.0; older versions '
             'never check for this field at all.',
    )
    p.add_argument(
        '--fix-subframes', metavar='DIR', default=None,
        help='Rebuild mdocs that are too short (restarted collections). '
             'Can be a directory (DIR/<stem>.txt is used for each mdoc) '
             'or a single .txt file (used directly, for one mdoc at a time). '
             'The .txt file lists the correct fraction filenames one per line; '
             'tilt and acquisition order are parsed from the filenames and '
             'all other metadata is copied from the existing mdoc. '
             'ExposureDose is also injected using --dose.',
    )
    p.add_argument(
        '--dose', type=float, default=4.16, metavar='DOSE',
        help='ExposureDose value to inject (e⁻/Å² per tilt; '
             'total dose for the whole exposure, not per frame). '
             'Default: 4.16 (8 sub-frames × 0.52 e⁻/Å² per frame, '
             'typical for K3 TIFF tilt series).  Used by --fix-dose '
             'and --fix-subframes.',
    )
    p.add_argument(
        '--exptime', type=float, default=1.0, metavar='SECONDS',
        help='ExposureTime value to inject when entirely absent (seconds; '
             'AreTomo3 only uses this to weight per-tilt dose RELATIVE to '
             'the series total, so an identical value in every section is '
             'safe and just yields equal weighting -- the same fallback '
             'AreTomo3 itself uses when the total is zero). Default: 1.0. '
             'Used by --fix-exptime.',
    )
    p.add_argument(
        '--check-frames', dest='check_frames', action='store_true', default=True,
        help='Verify every SubFramePath-referenced movie file actually '
             'exists next to the mdoc (AreTomo3 requires raw movies and '
             'their mdoc to be co-located). On by default; use '
             '--no-check-frames to skip (e.g. validating mdocs before '
             'movies finish transferring).',
    )
    p.add_argument(
        '--no-check-frames', dest='check_frames', action='store_false',
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        '--calibrated-apix', type=float, default=None, metavar='ANGPIX',
        help='Record the real, calibrated pixel size (Å/px) up front, at '
             'the start of the pipeline. The mdoc\'s own PixelSpacing is '
             'uncalibrated and routinely wrong; every later --apix/--angpix '
             'preflight check (run-aretomo3, run-aretomo3-per-ts, analyse) '
             'compares against this value instead once it is set, instead '
             'of nagging against the raw mdoc value on every run.',
    )
    p.set_defaults(func=run)
    return p


def run(args):
    paths = args.mdoc_files
    fix_order = args.fix_order
    fix_dose = args.fix_dose
    fix_exptime = args.fix_exptime
    fix_subframes = args.fix_subframes
    any_fix = fix_order or fix_dose or fix_exptime or fix_subframes
    n_pass = n_fail = n_fixed = 0
    n_fail_dose = n_fail_exptime = n_fail_order = n_fail_short = n_fail_frames = 0
    col_w = max(len(Path(p).name) for p in paths)

    print()
    print('{:<{w}}  {:6}  {:5}  {}'.format(
        'File', 'Result', 'Tilts', 'Notes', w=col_w))
    print('-' * (col_w + 60))

    passing_paths = []   # collect files that are clean for project.json save

    for path in paths:
        fname = Path(path).name
        r = validate_file(path, fix_dose=fix_dose, fix_exptime=fix_exptime,
                          fix_order=fix_order, fix_subframes=fix_subframes,
                          dose=args.dose, exptime=args.exptime,
                          check_frames=args.check_frames)

        if r['success'] and not r.get('fixed'):
            status = 'PASS'
            n_pass += 1
            passing_paths.append(path)
        elif r['success'] and r.get('fixed'):
            status = 'FIXED'
            n_fixed += 1
            n_pass += 1
            passing_paths.append(path)
        else:
            status = 'FAIL'
            n_fail += 1
            if r.get('n_sections', _MIN_TILTS) < _MIN_TILTS:
                n_fail_short += 1
            if r.get('failure') == 'missing_exposuredose':
                n_fail_dose += 1
            if r.get('failure') == 'missing_exposuretime':
                n_fail_exptime += 1
            if r.get('order_issues'):
                n_fail_order += 1
            if r.get('n_frames_found') is not None and r['n_frames_found'] < r['n_frames_expected']:
                n_fail_frames += 1

        note_parts = []
        if r['failure'] and not r['fixed']:
            note_parts.append(_failure_message(r['failure'], r))
        if r['fixed']:
            fix_types = r.get('fix_types', [])
            fix_descriptions = []
            if 'subframes' in fix_types:
                fix_descriptions.append(
                    'rebuilt from SubFramePaths ({} sections)'.format(r['n_injected'])
                )
            if 'order' in fix_types:
                fix_descriptions.append('reordered fields')
            if 'dose' in fix_types:
                fix_descriptions.append('injected ExposureDose={}'.format(args.dose))
            if 'exptime' in fix_types:
                fix_descriptions.append('injected ExposureTime={}'.format(args.exptime))
            note_parts.append(
                '; '.join(fix_descriptions) +
                '; backup: {}.original.bak'.format(fname)
            )
        note = '; '.join(note_parts) if note_parts else ''

        print('{:<{w}}  {:6}  {:5d}  {}'.format(
            fname, status, r['n_tilts'], note, w=col_w))

        # Only show the pre-fix diagnostic issues when the file is still
        # failing -- once fixed, r['issues'] still contains the original
        # description (kept for auditing) but showing it under a FIXED row
        # reads as if the problem is still present.
        if status == 'FAIL':
            for issue in r.get('issues', []):
                print('  {:<{w}}         └─ {}'.format('', issue, w=col_w))

    print()
    if any_fix:
        print('Summary: {} already valid, {} fixed, {} could not be fixed  '
              '(out of {} files)'.format(
                  n_pass - n_fixed, n_fixed, n_fail, len(paths)))
    else:
        print('Summary: {} PASS, {} FAIL  (out of {} files)'.format(
            n_pass, n_fail, len(paths)))

        if n_fail > 0:
            if n_fail_order:
                print('  {:3d} field(s) out of order  → fix with --fix-order '
                      '(no value needed)'.format(n_fail_order))
            if n_fail_dose:
                print('  {:3d} missing ExposureDose  → fix with --fix-dose '
                      '--dose <value>'.format(n_fail_dose))
            if n_fail_exptime:
                print('  {:3d} missing ExposureTime  → fix with --fix-exptime '
                      '--exptime <value>'.format(n_fail_exptime))
            if n_fail_short:
                print('  {:3d} too short (< {} sections)'.format(
                    n_fail_short, _MIN_TILTS))
                print()
                print('  NOTE: short mdocs may result from a restarted data collection')
                print('  where the acquisition ran to completion but the mdoc was')
                print('  overwritten with only a few sections.  If the movie files')
                print('  exist on disk, the mdoc can be rebuilt:')
                print('    aretomo3-preprocess validate-mdoc <files> --fix-subframes <dir>')
                print('  where <dir> contains a .txt file per mdoc listing the correct')
                print('  fraction filenames (one per line).')
            if n_fail_frames:
                print('  {:3d} referenced movie file(s) not found next to the mdoc '
                      '(see └─ lines above)'.format(n_fail_frames))
    print()

    # ── Save mdoc metadata to project.json for all passing files ──────────────
    # Merges into existing mdoc_data so running on a subset (e.g. to fix a
    # few files) updates only those entries without wiping the rest.
    if passing_paths:
        _save_mdoc_to_project(passing_paths)

    if args.calibrated_apix is not None:
        from aretomo3_preprocess.shared.project_state import record_calibrated_apix
        record_calibrated_apix(args.calibrated_apix, source='validate-mdoc (user-supplied)')
        print(f'  calibrated_apix : {args.calibrated_apix} Å/px -- later --apix/--angpix '
              f'preflight checks will compare against this instead of the raw mdoc value')


def _save_mdoc_to_project(paths):
    """
    Parse validated mdoc files and save rich metadata to project.json,
    merging with any previously saved entries so that running on a subset
    (e.g. to fix a few files) updates only those entries without wiping
    data for the rest of the dataset.

    Thin wrapper over shared/project_state.py:register_mdoc_data() -- see
    its docstring for the key-resolution strategy (path.resolve().stem,
    correct for both a renamed ts-XXXX.mdoc symlink and an original
    Position_N.mdoc passed directly, no rename_ts.lookup dependency
    needed) and the stale-key-stripping/frames_dir behavior this shares
    with enrich.py's --mdoc-data handler (previously two independent,
    drifted implementations -- see CLAUDE.md).
    """
    try:
        from aretomo3_preprocess.shared.project_state import register_mdoc_data
    except ImportError:
        return

    result = register_mdoc_data(paths)
    if result['n_ok']:
        print(f'  mdoc_data: {result["n_ok"]} TS saved'
              + (f' ({result["merged_count"]} total in project.json)'
                 if result['merged_count'] > result['n_ok'] else ''))


def _failure_message(failure, r):
    msgs = {
        'missing_exposuredose': (
            'ExposureDose absent from {} section(s) — '
            'AreTomo3 parser stalls after TiltAngle'.format(
                len(r.get('missing_dose_sections', [])) or '?')
        ),
        'missing_tiltangle':    'TiltAngle missing from one or more sections',
        'missing_subframepath': 'SubFramePath missing from one or more sections',
        'missing_exposuretime': (
            'ExposureTime missing (or out of order relative to TiltAngle/'
            'ExposureDose/SubFramePath) in one or more sections — required '
            'by AreTomo3 >= 2.3.0; parser stalls after SubFramePath'
        ),
        'no_zvalue':            'No [ZValue = N] sections found',
        'too_few_tilts':        'Fewer than {} tilts loaded'.format(_MIN_TILTS),
    }
    return msgs.get(failure, 'Unknown failure: {}'.format(failure))
