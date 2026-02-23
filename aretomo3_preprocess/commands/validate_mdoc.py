"""
validate-mdoc — check and optionally repair SerialEM mdoc files for AreTomo3.

What it checks
--------------
AreTomo3 reads four fields from each tilt section using a strict sequential
state machine (CReadMdoc::DoIt):

    ZValue  →  TiltAngle  →  ExposureDose  →  SubFramePath

If *any* of these are absent the parser stalls and the whole file fails
(returns false, tilt series skipped).

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
  # Validate only — report issues and suggest fix command:
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc frames/ts-129.mdoc
  aretomo3-preprocess validate-mdoc frames/ts-*.mdoc

  # Apply fix — inject ExposureDose into sections where it is missing:
  aretomo3-preprocess validate-mdoc frames/ts-128.mdoc --fix --dose 0.52

  # Fix all failing files at once:
  aretomo3-preprocess validate-mdoc frames/ts-*.mdoc --fix --dose 0.52

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


# ─────────────────────────────────────────────────────────────────────────────
# Simulate CReadMdoc::DoIt
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_aretomo3(lines):
    """
    Simulate the AreTomo3 mdoc parser state machine.

    Returns (n_tilts_loaded, failure_state) where failure_state is one of:
      None                   — success
      'no_zvalue'            — no [ZValue = N] sections found
      'missing_tiltangle'    — stuck waiting for TiltAngle
      'missing_exposuredose' — stuck waiting for ExposureDose
      'missing_subframepath' — stuck waiting for SubFramePath
      'too_few_tilts'        — parsed OK but fewer than 7 tilts
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
            elif bFm:
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
        issues=issues,
    )


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
# Fix: rebuild mdoc from a SubFramePaths file (too-short / restarted series)
# ─────────────────────────────────────────────────────────────────────────────

_SFNAME_RE = re.compile(
    r'^(.+_(\d{3})_([-\d.]+)_\d{8}_\d{6}_fractions\.tiff)\s*$',
    re.IGNORECASE,
)


def _parse_subframes_file(path):
    """
    Read a SubFramePaths .txt file (one filename per line).
    Returns a list of (acq_order, tilt_angle, filename) sorted by acq_order.
    Filename format: <PositionName>_NNN_TILT_DATE_TIME_fractions.tiff
    """
    entries = []
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        m = _SFNAME_RE.match(line.strip())
        if m:
            entries.append((int(m.group(2)), float(m.group(3)), m.group(1)))
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

def validate_file(path, fix_dose=False, fix_subframes=None, dose=4.16):
    """
    Validate one mdoc file.  Optionally apply fixes.

    fix_dose      — inject missing ExposureDose (--fix-dose)
    fix_subframes — path to a directory of SubFramePaths .txt files;
                    used to rebuild mdocs that are too short (--fix-subframes)

    Returns a dict with keys:
      path, success, n_tilts, failure, issues, fixed, fix_type,
      n_injected, backed_up
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        return dict(path=path, success=False, n_tilts=0,
                    failure=str(e), issues=[], fixed=False, fix_type=None,
                    n_injected=0, backed_up=False)

    text = raw.decode('latin-1', errors='replace')
    lines = [l + '\n' for l in text.splitlines()]

    n_tilts, failure = _simulate_aretomo3(lines)
    scan = _scan_fields(lines)

    result = dict(
        path=path,
        success=(failure is None),
        n_tilts=n_tilts,
        failure=failure,
        issues=scan['issues'],
        missing_dose_sections=scan['missing_dose_sections'],
        fixed=False,
        fix_type=None,
        n_injected=0,
        backed_up=False,
    )

    if failure is None:
        return result  # already valid

    # ── Fix subframes (too-short / restarted series) ────────────────────────
    if fix_subframes is not None and scan['n_sections'] < _MIN_TILTS:
        stem = Path(path).stem
        sf_path = Path(fix_subframes) / '{}.txt'.format(stem)
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
        if failure_after is not None:
            result['issues'].append(
                'Rebuilt mdoc still fails AreTomo3 simulation ({}) '
                '— file NOT written'.format(failure_after)
            )
            return result

        Path(path).write_text(''.join(new_lines), encoding='latin-1')
        result['fixed'] = True
        result['fix_type'] = 'subframes'
        result['n_injected'] = n_sections
        result['success'] = True
        result['n_tilts'] = n_after
        result['failure'] = None
        return result

    # ── Fix dose (missing ExposureDose) ─────────────────────────────────────
    if fix_dose:
        if failure != 'missing_exposuredose':
            result['issues'].append(
                'Cannot auto-fix failure type "{}"; '
                '--fix-dose only repairs missing ExposureDose'.format(failure)
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

        new_lines, n_injected = _inject_dose(lines, dose)

        n_after, failure_after = _simulate_aretomo3(new_lines)
        if failure_after is not None:
            result['issues'].append(
                'Fix did not resolve parse failure ({}) — file NOT written'.format(
                    failure_after)
            )
            return result

        Path(path).write_text(''.join(new_lines), encoding='latin-1')
        result['fixed'] = True
        result['fix_type'] = 'dose'
        result['n_injected'] = n_injected
        result['success'] = True
        result['n_tilts'] = n_after
        result['failure'] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'validate-mdoc',
        help='Validate SerialEM mdoc files for AreTomo3 compatibility; '
             'optionally inject missing ExposureDose',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        'mdoc_files', nargs='+', metavar='MDOC',
        help='One or more .mdoc files to validate (glob expansion handled by shell)',
    )
    p.add_argument(
        '--fix-dose', action='store_true', default=False,
        help='Inject missing ExposureDose into failing files.',
    )
    p.add_argument(
        '--fix-subframes', metavar='DIR', default=None,
        help='Rebuild mdocs that are too short (restarted collections) using '
             'SubFramePaths .txt files in DIR.  For each failing mdoc, '
             'DIR/<stem>.txt must list the correct fraction filenames '
             '(one per line).  Tilt and acquisition order are parsed from '
             'the filenames; all other metadata is copied from the existing '
             'mdoc.  ExposureDose is also injected using --dose.',
    )
    p.add_argument(
        '--dose', type=float, default=4.16, metavar='DOSE',
        help='ExposureDose value to inject (e⁻/Å² per tilt; '
             'total dose for the whole exposure, not per frame). '
             'Default: 4.16 (8 sub-frames × 0.52 e⁻/Å² per frame, '
             'typical for K3 TIFF tilt series).  Used by both --fix-dose '
             'and --fix-subframes.',
    )
    p.set_defaults(func=run)
    return p


def run(args):
    paths = args.mdoc_files
    fix_dose = args.fix_dose
    fix_subframes = args.fix_subframes
    n_pass = n_fail = n_fixed = 0
    col_w = max(len(Path(p).name) for p in paths)

    print()
    print('{:<{w}}  {:6}  {:5}  {}'.format(
        'File', 'Result', 'Tilts', 'Notes', w=col_w))
    print('-' * (col_w + 60))

    for path in paths:
        fname = Path(path).name
        r = validate_file(path, fix_dose=fix_dose,
                          fix_subframes=fix_subframes, dose=args.dose)

        if r['success'] and not r.get('fixed'):
            status = 'PASS'
            n_pass += 1
        elif r['success'] and r.get('fixed'):
            status = 'FIXED'
            n_fixed += 1
            n_pass += 1
        else:
            status = 'FAIL'
            n_fail += 1

        note_parts = []
        if r['failure'] and not r['fixed']:
            note_parts.append(_failure_message(r['failure'], r))
        if r['fixed']:
            if r.get('fix_type') == 'subframes':
                note_parts.append(
                    'rebuilt from SubFramePaths ({} sections); '
                    'backup: {}.original.bak'.format(r['n_injected'], fname)
                )
            else:
                note_parts.append(
                    'injected ExposureDose={} into {} section(s); '
                    'backup: {}.original.bak'.format(
                        args.dose, r['n_injected'], fname)
                )
        note = '; '.join(note_parts) if note_parts else ''

        print('{:<{w}}  {:6}  {:5d}  {}'.format(
            fname, status, r['n_tilts'], note, w=col_w))

        for issue in r.get('issues', []):
            print('  {:<{w}}         └─ {}'.format('', issue, w=col_w))

    print()
    if fix_dose or fix_subframes:
        print('Summary: {} already valid, {} fixed, {} could not be fixed  '
              '(out of {} files)'.format(
                  n_pass - n_fixed, n_fixed, n_fail, len(paths)))
    else:
        print('Summary: {} PASS, {} FAIL  (out of {} files)'.format(
            n_pass, n_fail, len(paths)))

        if n_fail > 0:
            print()
            print('To fix files with missing ExposureDose:')
            print('  aretomo3-preprocess validate-mdoc <files> --fix-dose')
            print('To rebuild too-short mdocs from a SubFramePaths directory:')
            print('  aretomo3-preprocess validate-mdoc <files> --fix-subframes <dir>')
    print()


def _failure_message(failure, r):
    msgs = {
        'missing_exposuredose': (
            'ExposureDose absent from {} section(s) — '
            'AreTomo3 parser stalls after TiltAngle'.format(
                len(r.get('missing_dose_sections', [])) or '?')
        ),
        'missing_tiltangle':    'TiltAngle missing from one or more sections',
        'missing_subframepath': 'SubFramePath missing from one or more sections',
        'no_zvalue':            'No [ZValue = N] sections found',
        'too_few_tilts':        'Fewer than {} tilts loaded'.format(_MIN_TILTS),
    }
    return msgs.get(failure, 'Unknown failure: {}'.format(failure))
