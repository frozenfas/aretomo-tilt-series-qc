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
    import mdocfile as _mdocfile
    _HAS_MDOCFILE = True
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


# ─────────────────────────────────────────────────────────────────────────────
# mdoc parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_mdoc_file(filepath):
    """
    Parse a SerialEM .mdoc file using the mdocfile library.

    Returns a dict keyed by ZValue (0-indexed acquisition order):
        {'sub_frame_path', 'mdoc_defocus', 'target_defocus', 'datetime',
         'stage_x', 'stage_y', 'stage_z', 'exposure_time', 'num_subframes'}
    Returns empty dict if mdocfile is not installed.
    """
    if not _HAS_MDOCFILE:
        return {}
    df = _mdocfile.read(filepath)
    result = {}
    for _, row in df.iterrows():
        z = _int_or_none(row.get('ZValue'))
        if z is None:
            continue
        sub = row.get('SubFramePath', None)
        stage = row.get('StagePosition', None)
        result[z] = {
            'sub_frame_path': Path(sub).name if sub and not isinstance(sub, float) else None,
            'mdoc_defocus':   _float_or_none(row.get('Defocus')),
            'target_defocus': _float_or_none(row.get('TargetDefocus')),
            'datetime':       row.get('DateTime') or None,
            'stage_x':        float(stage[0]) if stage and not isinstance(stage, float) else None,
            'stage_y':        float(stage[1]) if stage and not isinstance(stage, float) else None,
            'stage_z':        _float_or_none(row.get('StageZ')),
            'exposure_time':  _float_or_none(row.get('ExposureTime')),
            'num_subframes':  _int_or_none(row.get('NumSubFrames')),
        }
    return result
