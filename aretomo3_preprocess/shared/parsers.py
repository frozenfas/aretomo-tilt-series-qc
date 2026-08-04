"""
Parsers for AreTomo3 and SerialEM file formats.

    parse_aln_file   — AreTomo .aln alignment file
    parse_ctf_file   — AreTomo *_CTF.txt CTF estimates
    parse_tlt_file   — AreTomo *_TLT.txt tilt/dose table
    parse_mdoc_file  — SerialEM .mdoc metadata file
"""

import re
import numpy as np
from pathlib import Path

try:
    import logging as _logging
    import mdocfile as _mdocfile
    _HAS_MDOCFILE = True
    _logging.getLogger('mdocfile').setLevel(_logging.ERROR)
except ImportError:
    _HAS_MDOCFILE = False


def _float_or_none(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _int_or_none(v):
    try:
        f = float(v)
        return None if np.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# .aln parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_aln_file(filepath):
    """
    Parse one AreTomo .aln file.

    Returns a dict with:
        width, height, total_frames          – from RawSize header
        alpha_offset, beta_offset            – stage tilt offsets
        thickness                            – reconstructed thickness (px)
        num_patches                          – number of local-alignment patches
        dark_frames  : list of dicts         – {frame_a, frame_b, tilt}
        frames       : list of dicts         – {sec, rot, gmag, tx, ty,
                                                smean, sfit, scale, base, tilt}

    frame_b (dark_frames) and sec (frames) are AreTomo3's own 1-indexed SEC
    numbers in the tilt-sorted stack — the exact, canonical key for
    cross-referencing a frame against IMOD's order_list.csv/newstack section
    numbers, _TLT.txt rows, etc. Use them directly; don't re-derive a match
    by comparing tilt angles or anything else approximate (see CLAUDE.md,
    "Cross-referencing frames").
    """
    width = height = total_frames = None
    alpha_offset = beta_offset = thickness = num_patches = None
    dark_frames, frames = [], []

    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith('#'):
                m = re.match(r'#\s*RawSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)', line)
                if m:
                    width, height, total_frames = int(m[1]), int(m[2]), int(m[3])
                    continue

                m = re.match(r'#\s*AlphaOffset\s*=\s*([-\d.]+)', line)
                if m:
                    alpha_offset = float(m[1]); continue

                m = re.match(r'#\s*BetaOffset\s*=\s*([-\d.]+)', line)
                if m:
                    beta_offset = float(m[1]); continue

                m = re.match(r'#\s*Thickness\s*=\s*(\d+)', line)
                if m:
                    thickness = int(m[1]); continue

                m = re.match(r'#\s*NumPatches\s*=\s*(\d+)', line)
                if m:
                    num_patches = int(m[1]); continue

                # DarkFrame =  frame_a  frame_b  tilt_angle
                m = re.match(r'#\s*DarkFrame\s*=\s+(\d+)\s+(\d+)\s+([-\d.]+)', line)
                if m:
                    dark_frames.append({
                        'frame_a': int(m[1]),
                        'frame_b': int(m[2]),
                        'tilt':    float(m[3]),
                    })
                    continue

            else:
                # Data row: SEC  ROT  GMAG  TX  TY  SMEAN  SFIT  SCALE  BASE  TILT
                parts = line.split()
                if len(parts) == 10:
                    try:
                        frames.append({
                            'sec':   int(parts[0]),
                            'rot':   float(parts[1]),
                            'gmag':  float(parts[2]),
                            'tx':    float(parts[3]),
                            'ty':    float(parts[4]),
                            'smean': float(parts[5]),
                            'sfit':  float(parts[6]),
                            'scale': float(parts[7]),
                            'base':  float(parts[8]),
                            'tilt':  float(parts[9]),
                        })
                    except ValueError:
                        pass  # header row

    return {
        'width':        width,
        'height':       height,
        'total_frames': total_frames,
        'alpha_offset': alpha_offset,
        'beta_offset':  beta_offset,
        'thickness':    thickness,
        'num_patches':  num_patches,
        'dark_frames':  dark_frames,
        'frames':       frames,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_ctf_file(filepath):
    """
    Parse an AreTomo *_CTF.txt file.

    Columns: micrograph_number  defocus1_A  defocus2_A  astig_angle_deg
             phase_shift_rad  cc  fit_spacing_A  dfhand

    Returns a dict keyed by micrograph number (1-indexed).
    Mean defocus and astigmatism are added in both Å and µm.
    """
    ctf = {}
    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 8:
                try:
                    idx = int(parts[0])
                    d1  = float(parts[1])
                    d2  = float(parts[2])
                    ctf[idx] = {
                        'defocus1_A':       d1,
                        'defocus2_A':       d2,
                        'mean_defocus_A':   (d1 + d2) / 2.0,
                        'mean_defocus_um':  (d1 + d2) / 2.0 / 1e4,
                        'astig_A':          abs(d1 - d2),
                        'astig_um':         abs(d1 - d2) / 1e4,
                        'astig_angle_deg':  float(parts[3]),
                        'phase_shift_rad':  float(parts[4]),
                        'cc':               float(parts[5]),
                        'fit_spacing_A':    float(parts[6]),
                        'dfhand':           int(parts[7]),
                    }
                except ValueError:
                    pass
    return ctf


# ─────────────────────────────────────────────────────────────────────────────
# _TLT.txt parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_tlt_file(filepath):
    """
    Parse an AreTomo *_TLT.txt file.

    Each row N (1-indexed) corresponds to SEC N in the .aln / _CTF.txt files
    (tilt-sorted order, including dark frames).

    Returns a dict keyed by 1-indexed row number:
        {'nominal_tilt': float, 'acq_order': int,
         'dose_e_per_A2': float, 'z_value': int}
    where dose_e_per_A2 is the per-frame dose (not cumulative) and
    z_value = acq_order - 1  (0-indexed = ZValue in the mdoc file).
    """
    result = {}
    with open(filepath) as fh:
        for i, line in enumerate(fh, start=1):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    acq_order = int(parts[1])
                    result[i] = {
                        'nominal_tilt':  float(parts[0]),
                        'acq_order':     acq_order,
                        'dose_e_per_A2': float(parts[2]),
                        'z_value':       acq_order - 1,
                    }
                except ValueError:
                    pass
    return result


def check_nominal_tilt_consistency(sec_tilt_pairs, tlt_data, alpha_offset=0.0, tol=0.05):
    """
    Cross-check _TLT.txt's nominal_tilt against a tilt value from another
    source (.aln's TILT column, or a DarkFrame header line's tilt field)
    for the same SEC. _TLT.txt is always raw nominal regardless of
    -TiltCor; the other source already has alpha_offset baked in whenever
    -TiltCor produced a nonzero AlphaOffset (confirmed dataset-wide -- see
    CLAUDE.md's alpha_offset convention section) -- so the correct
    comparison is nominal_tilt + alpha_offset ≈ other_tilt, not plain
    equality. alpha_offset=0.0 (no TiltCor) reduces this to a plain
    equality check.

    sec_tilt_pairs: iterable of (sec, tilt) pairs to check, e.g.
        [(f['sec'], f['tilt']) for f in aln_data['frames']]
        [(df['frame_b'], df['tilt']) for df in aln_data['dark_frames']]
    tlt_data: {sec: {'nominal_tilt': ..., ...}} from parse_tlt_file.

    Returns a list of SEC numbers where the two disagree by more than tol
    degrees (SECs with no entry in tlt_data are skipped, not flagged).
    """
    alpha_offset = alpha_offset or 0.0
    bad = []
    for sec, tilt in sec_tilt_pairs:
        tlt = tlt_data.get(sec)
        if tlt is None:
            continue
        if abs(tlt['nominal_tilt'] + alpha_offset - tilt) > tol:
            bad.append(sec)
    return bad


def compute_reference_defocus(ctf_dir) -> dict:
    """
    Parse every ts-*_CTF.txt (+ matching ts-*_TLT.txt) in ctf_dir and
    return {ts_name: reference_defocus_um} -- the first-acquired tilt's
    (acq_order==1, from _TLT.txt) mean_defocus_um from _CTF.txt, or the
    median defocus across all fitted frames if _TLT.txt is missing or that
    specific frame wasn't fit.

    Computed fresh from files on disk every call -- deliberately NOT
    cached in project.json. Defocus estimates change whenever AreTomo3/
    CTFFIND is re-run, and a project.json cache has no way to know it's
    gone stale (this replaced an earlier project.json-cached defocus_data
    section, removed for exactly this reason -- see CLAUDE.md and
    imod_mtffilter.py, its only consumer). ctf_dir is normally a command's
    own already-required --input directory (e.g. imod-mtffilter's), not a
    separate registration step.

    TS with no parseable _CTF.txt rows are silently omitted from the
    result, not raised -- callers report as they see fit.
    """
    from pathlib import Path
    ctf_dir = Path(ctf_dir)
    per_ts = {}
    for ctf_path in sorted(ctf_dir.glob('ts-*_CTF.txt')):
        ts_name = ctf_path.stem[:-len('_CTF')]
        tlt_path = ctf_dir / f'{ts_name}_TLT.txt'
        try:
            ctf_data = parse_ctf_file(ctf_path)
            if not ctf_data:
                continue

            ref_sec = None
            if tlt_path.exists():
                tlt_data = parse_tlt_file(tlt_path)
                ref_sec = next(
                    (sec for sec, t in tlt_data.items() if t['acq_order'] == 1),
                    None,
                )
            defocus = ctf_data[ref_sec]['mean_defocus_um'] if ref_sec in ctf_data else None
            if defocus is None:
                vals = sorted(f['mean_defocus_um'] for f in ctf_data.values())
                defocus = vals[len(vals) // 2]

            per_ts[ts_name] = round(defocus, 4)
        except Exception:
            continue
    return per_ts


# ─────────────────────────────────────────────────────────────────────────────
# mdoc parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_mdoc_file(filepath):
    """
    Parse a SerialEM .mdoc file using the mdocfile library.

    Returns a tuple (frames, pixel_spacing, acquisition) where:
      frames        — dict keyed by ZValue (0-indexed acquisition order):
                        {'tilt_angle', 'sub_frame_path', 'nominal_defocus',
                         'target_defocus', 'datetime', 'stage_x/y/z',
                         'image_shift_x/y', 'exposure_time', 'num_subframes',
                         'exposure_dose'}
                        'exposure_dose' is the mdoc's own ExposureDose field
                        (e/Å², per this tilt's exposure/frame-set) -- distinct
                        from `parse_tlt_file`'s dose_e_per_A2, which is
                        AreTomo3's own per-frame dose from _TLT.txt.

                        'nominal_defocus' (mdoc's Defocus field) and
                        'target_defocus' (mdoc's TargetDefocus field) are
                        both straight from SerialEM, never touched by
                        AreTomo3/CTFFIND -- the same "nominal" category as
                        `parse_tlt_file`'s nominal_tilt, not a measurement.
                        The actual measured defocus for a frame is
                        `parse_ctf_file`'s mean_defocus_um (CTFFIND's fit,
                        via AreTomo3's _CTF.txt) -- keep these three
                        conceptually separate; see CLAUDE.md's alpha_offset
                        convention section for why this nominal-vs-measured
                        distinction matters (the same class of bug already
                        found once for tilt).
      pixel_spacing — float (Å/px) from the first row's PixelSpacing field,
                        or None if not present.
      acquisition   — {'width', 'height', 'file_type', 'voltage'} from the
                        first row's ImageSize/SubFramePath/Voltage fields
                        (each None if not present). 'file_type' is the raw
                        movie extension (e.g. 'tiff', 'eer'), lowercased,
                        without the dot.
    Returns ({}, None, {'width': None, 'height': None, 'file_type': None,
    'voltage': None}) if mdocfile is not installed.
    """
    _empty_acq = {'width': None, 'height': None, 'file_type': None, 'voltage': None}
    if not _HAS_MDOCFILE:
        return {}, None, dict(_empty_acq)
    df = _mdocfile.read(filepath)
    # Extract global PixelSpacing from first row (same value repeated in all rows)
    try:
        pixel_spacing = _float_or_none(df['PixelSpacing'].iloc[0])
    except Exception:
        pixel_spacing = None
    acquisition = dict(_empty_acq)
    try:
        size = df['ImageSize'].iloc[0]
        acquisition['width']  = _int_or_none(size[0])
        acquisition['height'] = _int_or_none(size[1])
    except Exception:
        pass
    try:
        sub0 = df['SubFramePath'].iloc[0]
        if sub0 and not isinstance(sub0, float):
            acquisition['file_type'] = Path(sub0).suffix.lstrip('.').lower() or None
    except Exception:
        pass
    try:
        acquisition['voltage'] = _float_or_none(df['Voltage'].iloc[0])
    except Exception:
        pass
    result = {}
    for _, row in df.iterrows():
        z = _int_or_none(row.get('ZValue'))
        if z is None:
            continue
        sub = row.get('SubFramePath', None)
        stage = row.get('StagePosition', None)
        img_shift = row.get('ImageShift', None)
        result[z] = {
            'tilt_angle':     _float_or_none(row.get('TiltAngle')),
            'sub_frame_path': Path(sub).name if sub and not isinstance(sub, float) else None,
            'nominal_defocus': _float_or_none(row.get('Defocus')),
            'target_defocus': _float_or_none(row.get('TargetDefocus')),
            'datetime':       row.get('DateTime') or None,
            'stage_x':        float(stage[0]) if stage and not isinstance(stage, float) else None,
            'stage_y':        float(stage[1]) if stage and not isinstance(stage, float) else None,
            'stage_z':        _float_or_none(row.get('StageZ')),
            'image_shift_x':  float(img_shift[0]) if img_shift and not isinstance(img_shift, float) else None,
            'image_shift_y':  float(img_shift[1]) if img_shift and not isinstance(img_shift, float) else None,
            'exposure_time':  _float_or_none(row.get('ExposureTime')),
            'num_subframes':  _int_or_none(row.get('NumSubFrames')),
            'exposure_dose':  _float_or_none(row.get('ExposureDose')),
        }
    return result, pixel_spacing, acquisition
