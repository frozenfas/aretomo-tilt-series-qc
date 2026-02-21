#!/usr/bin/env python3
"""
parse_aln.py  —  Parse AreTomo .aln / _CTF.txt files, quantify per-frame
                  misalignment and produce diagnostic PNG plots + HTML viewer.

Usage:
    python parse_aln.py --input run001/ --output run001_analysis/ --threshold 80
"""

import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

try:
    import mdocfile as _mdocfile
    _HAS_MDOCFILE = True
except ImportError:
    _HAS_MDOCFILE = False


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
# _TLT.txt and mdoc parsing
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap(tx, ty, w, h):
    """
    % overlap of a frame shifted by (tx, ty) with the reference frame.

    All frames within a tilt series share the same ROT value, so the
    relative displacement between any frame and the reference is purely
    translational in the aligned coordinate system.  The overlap of two
    identically-rotated, same-sized rectangles separated by (tx, ty) is:

        overlap_x = max(0, W - |tx|) / W
        overlap_y = max(0, H - |ty|) / H
        % overlap  = overlap_x * overlap_y * 100
    """
    ox = max(0.0, w - abs(tx)) / w
    oy = max(0.0, h - abs(ty)) / h
    return ox * oy * 100.0


def rotated_rect_corners(cx, cy, w, h, angle_deg):
    """
    Return the 4 corners of a W×H rectangle centred at (cx, cy),
    rotated in-plane by angle_deg.
    """
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in local]


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

# Overlap colourmap: red (0%) → green (100%)
OVL_CMAP = plt.cm.RdYlGn
OVL_NORM = plt.Normalize(vmin=0, vmax=100)


def _ovl_colour(overlap_pct):
    return OVL_CMAP(OVL_NORM(overlap_pct))


def _ovl_sm():
    sm = plt.cm.ScalarMappable(cmap=OVL_CMAP, norm=OVL_NORM)
    sm.set_array([])
    return sm


# Resolution colourmap: green (good/low Å) → red (poor/high Å)
# fit_spacing_A: lower = better resolution → we want low values to be green
# Using RdYlGn_r: value=0 → green, value=1 → red; normalise so low Å → 0
RES_CMAP = plt.cm.RdYlGn_r


def _res_sm(vmin, vmax):
    sm = plt.cm.ScalarMappable(cmap=RES_CMAP,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm


# ─────────────────────────────────────────────────────────────────────────────
# Per-tilt-series plot  (2 × 2 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tilt_series(ts_name, data, threshold, out_path, global_ranges):
    frames      = data['frames']
    W, H        = data['width'], data['height']
    dark_frames = data['dark_frames']

    tilts    = np.array([f['tilt']         for f in frames])
    overlaps = np.array([f['overlap_pct']  for f in frames])
    is_ref   = np.array([f['is_reference'] for f in frames])
    rot      = frames[0]['rot'] if frames else 0.0

    dark_tilts = [df['tilt'] for df in dark_frames]
    n_bad      = int(np.sum(overlaps < threshold))
    ovl_cols   = [_ovl_colour(o) for o in overlaps]
    ovl_sm     = _ovl_sm()

    has_ctf  = any(f.get('mean_defocus_um') is not None for f in frames)
    res_sm   = _res_sm(global_ranges['fit_spacing_min'],
                       global_ranges['fit_spacing_max'])

    # Global axis limits (consistent across all tilt series)
    tilt_xlim   = (global_ranges['tilt_min'],    global_ranges['tilt_max'])
    defocus_ylim = (global_ranges['defocus_min'], global_ranges['defocus_max'])

    fig = plt.figure(figsize=(20, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f'{ts_name}   |   {len(frames)} aligned frames   |   '
        f'{len(dark_frames)} dark frames   |   '
        f'{n_bad} frame(s) below {threshold}% overlap threshold',
        fontsize=12, fontweight='bold',
    )

    # ── Panel 1 : Overlap % vs tilt angle ────────────────────────────────────
    for i in range(len(frames)):
        mk = '*' if is_ref[i] else 'o'
        sz = 160 if is_ref[i] else 55
        ec = 'black' if is_ref[i] else 'none'
        ax1.scatter(tilts[i], overlaps[i], color=ovl_cols[i], s=sz,
                    marker=mk, edgecolors=ec, linewidths=0.9, zorder=3)

    for dt in dark_tilts:
        ax1.axvline(dt, color='#cccccc', lw=0.8, zorder=1)

    ax1.axhline(threshold, color='steelblue', lw=1.3, ls='--',
                label=f'{threshold}% threshold')
    plt.colorbar(ovl_sm, ax=ax1, label='% Overlap')
    ax1.set_xlim(*tilt_xlim)
    ax1.set_ylim(-5, 108)
    ax1.set_xlabel('Corrected tilt angle (°)')
    ax1.set_ylabel('% Overlap with reference frame')
    ax1.set_title('Overlap vs Corrected Tilt Angle')
    ax1.legend(fontsize=8)

    # ── Panel 2 : Spatial rectangle plot ─────────────────────────────────────
    all_x, all_y = [], []

    for i, f in enumerate(frames):
        if is_ref[i]:
            continue
        alpha   = 0.22 if overlaps[i] >= threshold else 0.50
        corners = rotated_rect_corners(f['tx'], f['ty'], W, H, rot)
        poly    = plt.Polygon(corners, closed=True,
                              facecolor=ovl_cols[i], alpha=alpha,
                              edgecolor=ovl_cols[i], linewidth=0.4, zorder=2)
        ax2.add_patch(poly)
        xs, ys = zip(*corners)
        all_x.extend(xs); all_y.extend(ys)

    ref_corners = rotated_rect_corners(0, 0, W, H, rot)
    ref_poly    = plt.Polygon(ref_corners, closed=True,
                              fill=False, edgecolor='black',
                              linewidth=2.5, zorder=5)
    ax2.add_patch(ref_poly)
    ax2.scatter(0, 0, color='black', s=90, marker='*', zorder=6)
    xs, ys = zip(*ref_corners)
    all_x.extend(xs); all_y.extend(ys)

    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        xpad = (xmax - xmin) * 0.05 or W * 0.05
        ypad = (ymax - ymin) * 0.05 or H * 0.05
        ax2.set_xlim(xmin - xpad, xmax + xpad)
        ax2.set_ylim(ymin - ypad, ymax + ypad)

    plt.colorbar(ovl_sm, ax=ax2, label='% Overlap')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('X offset (px)')
    ax2.set_ylabel('Y offset (px)')
    ax2.set_title(f'Frame Positions  (in-plane rot = {rot:.1f}°)')
    ax2.legend(
        handles=[Line2D([0], [0], color='black', lw=2.5, label='Reference frame')],
        fontsize=8,
    )

    # ── Panel 3 : Tilt coverage diagram ──────────────────────────────────────
    # Each frame at corrected tilt θ is a full line through the origin:
    #   A = (-cos θ, -sin θ)  →  B = (cos θ, sin θ)
    # θ = 0° → horizontal line (-1,0)–(1,0); 0° is to the right.
    # Viewing geometry: X axis = beam direction at 0°, Y axis = vertical.
    ax3.axhline(0, color='#555555', lw=0.8, zorder=1)   # sample plane at 0°
    ax3.axvline(0, color='#888888', lw=0.6, ls=':', zorder=1)

    # Dark frames: dashed grey lines through origin
    for dt in dark_tilts:
        tr  = np.radians(dt)
        ctr = np.cos(tr)
        str_ = np.sin(tr)
        ax3.plot([-ctr, ctr], [-str_, str_],
                 color='#bbbbbb', lw=1.0, ls='--', zorder=2)

    # Aligned frames: full lines through origin + right-endpoint dot
    for i in range(len(frames)):
        tr   = np.radians(tilts[i])
        ctr  = np.cos(tr)
        str_ = np.sin(tr)
        lw   = 2.8 if is_ref[i] else 1.6
        ax3.plot([-ctr, ctr], [-str_, str_],
                 color=ovl_cols[i], lw=lw, zorder=3)
        mk  = '*' if is_ref[i] else 'o'
        sz  = 110 if is_ref[i] else 35
        ec  = 'black' if is_ref[i] else 'none'
        # Small dot at right endpoint (cos θ, sin θ) to mark direction
        ax3.scatter(ctr, str_, color=ovl_cols[i],
                    s=sz, marker=mk, edgecolors=ec, linewidths=0.8, zorder=4)

    # Square symmetric limits — endpoints always on unit circle
    lim = 1.18
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    plt.colorbar(ovl_sm, ax=ax3, label='% Overlap')
    ax3.set_xlabel('cos(corrected tilt)')
    ax3.set_ylabel('sin(corrected tilt)')
    ax3.set_title('Tilt Coverage  (0° = horizontal,  -- = dark frame)')

    # ── Panel 4 : Defocus vs tilt angle ──────────────────────────────────────
    if has_ctf:
        def_um   = np.array([f.get('mean_defocus_um', np.nan) for f in frames])
        astig_um = np.array([f.get('astig_um',         0.0)   for f in frames])
        spacing  = np.array([f.get('fit_spacing_A',    np.nan) for f in frames])

        res_norm = plt.Normalize(vmin=global_ranges['fit_spacing_min'],
                                 vmax=global_ranges['fit_spacing_max'])
        res_cols = [RES_CMAP(res_norm(s)) if not np.isnan(s) else (0.7, 0.7, 0.7, 1)
                    for s in spacing]

        for i in range(len(frames)):
            if np.isnan(def_um[i]):
                continue
            mk = '*' if is_ref[i] else 'o'
            ec = 'black' if is_ref[i] else 'none'
            ax4.errorbar(tilts[i], def_um[i],
                         yerr=astig_um[i] / 2.0,
                         fmt='none', ecolor=res_cols[i], elinewidth=0.8,
                         capsize=2, zorder=2)
            ax4.scatter(tilts[i], def_um[i],
                        color=res_cols[i], s=55, marker=mk,
                        edgecolors=ec, linewidths=0.8, zorder=3)

        for dt in dark_tilts:
            ax4.axvline(dt, color='#cccccc', lw=0.8, zorder=1)

        plt.colorbar(res_sm, ax=ax4,
                     label='Estimated resolution — fit spacing (Å)\n'
                            'green = good (low Å),  red = poor (high Å)')
        ax4.set_xlim(*tilt_xlim)
        ax4.set_ylim(*defocus_ylim)
        ax4.set_xlabel('Corrected tilt angle (°)')
        ax4.set_ylabel('Mean defocus (µm)')
        ax4.set_title('Defocus vs Corrected Tilt Angle\n(error bars = astigmatism)')
    else:
        ax4.text(0.5, 0.5, 'No CTF data found\n(no _CTF.txt file)',
                 ha='center', va='center', transform=ax4.transAxes,
                 fontsize=10, color='grey')
        ax4.set_axis_off()

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Global summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_summary(all_ts, threshold, global_ranges, out_path):
    """
    Produce a 3×3 grid of histograms summarising all tilt series.

    Panels:
      (0,0) Frame counts per TS — n_total, n_aligned, n_passing
      (0,1) In-plane rotation ROT distribution
      (0,2) AlphaOffset (tilt axis correction) distribution
      (1,0) Mean defocus (µm) — all frames
      (1,1) Estimated resolution (fit_spacing_A) — all frames
      (1,2) % Overlap — all frames, with threshold vline
      (2,0) Astigmatism (µm) — all frames
      (2,1) Lamella thickness AlignZ (nm or px) per TS
      (2,2) Flagged frames per TS
    """
    ts_names = list(all_ts.keys())
    n_ts = len(ts_names)

    # Per-TS accumulators
    n_total_list      = []
    n_aligned_list    = []
    n_passing_list    = []
    n_flagged_list    = []
    rot_list          = []
    alpha_offset_list = []
    thickness_list    = []   # in nm if angpix known, else px
    thickness_unit    = 'px'
    thickness_angpix  = None

    # Per-frame accumulators
    defocus_all = []
    spacing_all = []
    overlap_all = []
    astig_all   = []

    for data in all_ts.values():
        frames = data['frames']
        n_total_list.append(data.get('total_frames') or len(frames))
        n_aligned_list.append(len(frames))
        n_flagged = sum(1 for f in frames if f['is_flagged'])
        n_flagged_list.append(n_flagged)
        n_passing_list.append(len(frames) - n_flagged)

        if frames:
            rot_list.append(frames[0]['rot'])

        if data.get('alpha_offset') is not None:
            alpha_offset_list.append(data['alpha_offset'])

        if data.get('thickness_nm') is not None:
            thickness_list.append(data['thickness_nm'])
            thickness_unit   = 'nm'
            thickness_angpix = data.get('angpix')
        elif data.get('thickness') is not None:
            thickness_list.append(data['thickness'])

        for f in frames:
            overlap_all.append(f['overlap_pct'])
            if f.get('mean_defocus_um') is not None:
                defocus_all.append(f['mean_defocus_um'])
            if f.get('fit_spacing_A') is not None:
                spacing_all.append(f['fit_spacing_A'])
            if f.get('astig_um') is not None:
                astig_all.append(f['astig_um'])

    def _median_vline(ax, data, color='steelblue'):
        """Draw a dashed median line; returns label string."""
        if not data:
            return
        med = np.median(data)
        ax.axvline(med, color=color, lw=1.5, ls='--',
                   label=f'median = {med:.2f}')

    def _no_data(ax, msg='No CTF data'):
        ax.text(0.5, 0.5, msg, ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='grey')
        ax.set_axis_off()

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle(f'Global Summary — {n_ts} Tilt Series',
                 fontsize=16, fontweight='bold')

    # ── (0,0) Frame counts per TS ─────────────────────────────────────────
    ax = axes[0, 0]
    all_counts = n_total_list + n_aligned_list + n_passing_list
    bins = np.arange(min(all_counts) - 0.5, max(all_counts) + 1.5, 1)
    ax.hist(n_total_list,   bins=bins, alpha=0.6, color='steelblue',
            label='Total (RawSize)')
    ax.hist(n_aligned_list, bins=bins, alpha=0.6, color='darkorange',
            label='Aligned')
    ax.hist(n_passing_list, bins=bins, alpha=0.6, color='forestgreen',
            label='Passing threshold')
    ax.set_xlabel('Frame count')
    ax.set_ylabel('# Tilt series')
    ax.set_title('Frame Counts per Tilt Series')
    ax.legend(fontsize=8)

    # ── (0,1) In-plane rotation ROT ───────────────────────────────────────
    ax = axes[0, 1]
    if rot_list:
        ax.hist(rot_list, bins=30, color='mediumpurple',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, rot_list)
        ax.set_xlabel('In-plane rotation ROT (°)')
        ax.set_ylabel('# Tilt series')
        ax.set_title('In-Plane Rotation Distribution')
        ax.legend(fontsize=8)
    else:
        _no_data(ax, 'No rotation data')

    # ── (0,2) AlphaOffset ─────────────────────────────────────────────────
    ax = axes[0, 2]
    if alpha_offset_list:
        ax.hist(alpha_offset_list, bins=30, color='teal',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, alpha_offset_list)
        ax.set_xlabel('AlphaOffset (°)')
        ax.set_ylabel('# Tilt series')
        ax.set_title('Tilt Axis Correction (AlphaOffset)')
        ax.legend(fontsize=8)
    else:
        _no_data(ax, 'No AlphaOffset data')

    # ── (1,0) Mean defocus ────────────────────────────────────────────────
    ax = axes[1, 0]
    if defocus_all:
        ax.hist(defocus_all, bins=50, color='cornflowerblue',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, defocus_all)
        ax.set_xlabel('Mean defocus (µm)')
        ax.set_ylabel('# Frames')
        ax.set_title('Mean Defocus Distribution')
        ax.legend(fontsize=8)
    else:
        _no_data(ax)

    # ── (1,1) Estimated resolution ────────────────────────────────────────
    ax = axes[1, 1]
    if spacing_all:
        ax.hist(spacing_all, bins=50, color='darkorange',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, spacing_all)
        ax.set_xlabel('Fit spacing (Å)')
        ax.set_ylabel('# Frames')
        ax.set_title('Estimated Resolution (fit_spacing_A)')
        ax.legend(fontsize=8)
    else:
        _no_data(ax)

    # ── (1,2) % Overlap ───────────────────────────────────────────────────
    ax = axes[1, 2]
    if overlap_all:
        ax.hist(overlap_all, bins=50, color='mediumseagreen',
                edgecolor='white', linewidth=0.3)
        ax.axvline(threshold, color='red', lw=1.5, ls='-.',
                   label=f'threshold = {threshold:.0f}%')
        _median_vline(ax, overlap_all)
        ax.set_xlabel('% Overlap with reference')
        ax.set_ylabel('# Frames')
        ax.set_title('Frame Overlap Distribution')
        ax.legend(fontsize=8)
    else:
        _no_data(ax, 'No overlap data')

    # ── (2,0) Astigmatism ─────────────────────────────────────────────────
    ax = axes[2, 0]
    if astig_all:
        ax.hist(astig_all, bins=50, color='salmon',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, astig_all)
        ax.set_xlabel('Astigmatism (µm)')
        ax.set_ylabel('# Frames')
        ax.set_title('Astigmatism Distribution')
        ax.legend(fontsize=8)
    else:
        _no_data(ax)

    # ── (2,1) Reconstruction thickness ───────────────────────────────────
    ax = axes[2, 1]
    if thickness_list:
        ax.hist(thickness_list, bins=30, color='slateblue',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, thickness_list)
        ax.set_xlabel(f'Lamella Thickness AlignZ ({thickness_unit})')
        ax.set_ylabel('# Tilt series')
        if thickness_unit == 'nm' and thickness_angpix is not None:
            title = f'Lamella Thickness AlignZ\n(pixel size = {thickness_angpix} Å/px)'
        elif thickness_unit == 'nm':
            title = 'Lamella Thickness AlignZ (nm)'
        else:
            title = 'Lamella Thickness AlignZ (px)\n(no pixel size — use --angpix)'
        ax.set_title(title)
        ax.legend(fontsize=8)
    else:
        _no_data(ax, 'No thickness data')

    # ── (2,2) Flagged frames per TS ───────────────────────────────────────
    ax = axes[2, 2]
    max_flagged = max(n_flagged_list) if n_flagged_list else 0
    bins_flag = np.arange(-0.5, max_flagged + 1.5, 1)
    ax.hist(n_flagged_list, bins=bins_flag, color='tomato',
            edgecolor='white', linewidth=0.3)
    mean_flagged = float(np.mean(n_flagged_list))
    ax.axvline(mean_flagged, color='steelblue', lw=1.5, ls='--',
               label=f'mean = {mean_flagged:.1f}')
    ax.set_xlabel('# Flagged frames')
    ax.set_ylabel('# Tilt series')
    ax.set_title(f'Flagged Frames per TS  (threshold = {threshold:.0f}%)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Global summary  : {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# HTML viewer
# ─────────────────────────────────────────────────────────────────────────────

def make_html(ts_entries, out_path, threshold):
    options_html = '\n'.join(
        f'    <option value="{i}" data-bad="{e["n_bad"]}">'
        f'{"[!] " if e["n_bad"] > 0 else "      "}'
        f'{e["name"]}  ({e["n_bad"]} flagged / {e["n_frames"]} frames)'
        f'</option>'
        for i, e in enumerate(ts_entries)
    )
    images_js = json.dumps([e['png'] for e in ts_entries])
    titles_js = json.dumps([
        f'{e["name"]}  —  {e["n_frames"]} aligned frames  |  '
        f'{e["n_dark"]} dark frames  |  {e["n_bad"]} below {threshold}% threshold'
        for e in ts_entries
    ])
    n = len(ts_entries)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AreTomo Alignment Analysis</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', sans-serif;
      background: #16213e;
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
      min-height: 100vh;
    }}
    h1 {{ margin-bottom: 18px; font-size: 1.25em; color: #90caf9; letter-spacing: 0.03em; }}
    #controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    button {{
      padding: 8px 20px;
      font-size: 1em;
      border: none;
      border-radius: 6px;
      background: #1565c0;
      color: white;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{ background: #0d47a1; }}
    select {{
      padding: 7px 10px;
      font-size: 0.88em;
      border-radius: 6px;
      background: #1e2a45;
      color: #e0e0e0;
      border: 1px solid #445;
      max-width: 380px;
      cursor: pointer;
    }}
    #counter {{ font-size: 0.88em; color: #90a4ae; min-width: 60px; text-align: center; }}
    #title {{
      font-size: 0.92em;
      color: #ffcc80;
      text-align: center;
      margin-bottom: 10px;
      min-height: 1.4em;
    }}
    #img-wrap {{
      width: 100%;
      max-width: 1380px;
      background: #0d1b2a;
      border-radius: 10px;
      padding: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    #main-img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    #hint {{ font-size: 0.75em; color: #546e7a; margin-top: 12px; }}
    #progress {{
      width: 100%;
      max-width: 1380px;
      height: 4px;
      background: #1e2a45;
      border-radius: 2px;
      margin-bottom: 10px;
    }}
    #progress-bar {{
      height: 100%;
      background: #1565c0;
      border-radius: 2px;
      transition: width 0.2s;
    }}
  </style>
</head>
<body>
  <h1>AreTomo Alignment Analysis</h1>

  <div id="controls">
    <button id="btn-prev">&#8592; Prev</button>
    <select id="ts-select">
{options_html}
    </select>
    <button id="btn-next">Next &#8594;</button>
    <span id="counter">1&nbsp;/&nbsp;{n}</span>
  </div>

  <div id="progress"><div id="progress-bar"></div></div>
  <div id="title"></div>

  <div id="img-wrap">
    <img id="main-img" src="" alt="tilt series plot">
  </div>

  <p id="hint">Keyboard: &#8592; &#8594; to navigate between tilt series</p>

  <script>
    const images  = {images_js};
    const titles  = {titles_js};
    const n       = images.length;
    let   idx     = 0;

    const img     = document.getElementById('main-img');
    const sel     = document.getElementById('ts-select');
    const ctr     = document.getElementById('counter');
    const ttl     = document.getElementById('title');
    const pbar    = document.getElementById('progress-bar');

    function show(i) {{
      idx = ((i % n) + n) % n;
      img.src          = images[idx];
      ttl.textContent  = titles[idx];
      ctr.innerHTML    = (idx + 1) + '&nbsp;/&nbsp;' + n;
      sel.value        = idx;
      pbar.style.width = ((idx + 1) / n * 100).toFixed(1) + '%';
    }}

    document.getElementById('btn-prev').onclick = () => show(idx - 1);
    document.getElementById('btn-next').onclick = () => show(idx + 1);
    sel.onchange = () => show(parseInt(sel.value));
    document.addEventListener('keydown', e => {{
      if (e.key === 'ArrowLeft')  show(idx - 1);
      if (e.key === 'ArrowRight') show(idx + 1);
    }});

    show(0);
  </script>
</body>
</html>
"""
    with open(out_path, 'w') as fh:
        fh.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Analyse AreTomo .aln alignment files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--input',     '-i', default='run001',
                    help='Directory containing .aln files')
    ap.add_argument('--output',    '-o', default='run001_analysis',
                    help='Output directory for plots, JSON, TSV, and HTML')
    ap.add_argument('--threshold', '-t', type=float, default=80.0,
                    help='%% overlap below which a frame is flagged')
    ap.add_argument('--mdocdir',   '-m', default='frames',
                    help='Directory containing per-TS .mdoc files '
                         '(ts-xxx.mdoc expected; skip enrichment if absent)')
    ap.add_argument('--angpix',   '-a', type=float, default=None,
                    help='Pixel size in Å/px — used to convert thickness from '
                         'pixels to nm.  If omitted, read from mdoc PixelSpacing.')
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    aln_files = sorted(in_dir.glob('*.aln'))
    if not aln_files:
        print(f'No .aln files found in {in_dir}')
        return

    mdoc_dir = Path(args.mdocdir)
    if not _HAS_MDOCFILE:
        print('WARNING: mdocfile not installed — mdoc enrichment skipped\n')
    elif not mdoc_dir.exists():
        print(f'WARNING: --mdocdir {mdoc_dir} not found — mdoc enrichment skipped\n')

    angpix_str = f'{args.angpix} Å/px' if args.angpix else 'from mdoc (or None)'
    print(f'Found {len(aln_files)} .aln files  |  threshold = {args.threshold}%  '
          f'|  pixel size = {angpix_str}\n')

    # ── Pass 1: parse all files, attach overlap + CTF + TLT + mdoc ───────────
    all_ts = {}   # ts_name → data dict

    for aln_path in aln_files:
        ts_name = aln_path.stem
        data    = parse_aln_file(aln_path)

        if not data['frames'] or data['width'] is None:
            print(f'  WARNING: {ts_name} could not be parsed — skipping')
            continue

        W, H = data['width'], data['height']

        # Load CTF (SEC N → CTF row N, 1-indexed, includes dark frames)
        ctf_path = aln_path.parent / f'{ts_name}_CTF.txt'
        ctf_data = parse_ctf_file(ctf_path) if ctf_path.exists() else {}

        # Load _TLT.txt (row N = SEC N, tilt-sorted, includes dark frames)
        tlt_path = aln_path.parent / f'{ts_name}_TLT.txt'
        tlt_data = parse_tlt_file(tlt_path) if tlt_path.exists() else {}

        # Load mdoc (keyed by ZValue = acq_order - 1)
        mdoc_path = mdoc_dir / f'{ts_name}.mdoc'
        mdoc_data = (parse_mdoc_file(mdoc_path)
                     if _HAS_MDOCFILE and mdoc_path.exists() else {})

        # Resolve pixel size: CLI arg → mdoc PixelSpacing → None
        angpix = args.angpix
        if angpix is None and mdoc_data:
            first_frame = next(iter(mdoc_data.values()))
            # PixelSpacing not stored in our mdoc dict; re-read from df if available
        if angpix is None and _HAS_MDOCFILE and mdoc_path.exists():
            try:
                _df = _mdocfile.read(mdoc_path)
                angpix = _float_or_none(_df['PixelSpacing'].iloc[0])
            except Exception:
                angpix = None
        data['angpix'] = angpix
        if data['thickness'] is not None and angpix is not None:
            data['thickness_nm'] = round(data['thickness'] * angpix / 10.0, 2)
        else:
            data['thickness_nm'] = None

        # Cumulative dose per acq_order (RELION convention: prior dose,
        # so the first acquired frame has cumulative = 0).
        # Sorted by acq_order; running sum accumulates dose of preceding frames.
        cum_dose_by_acq = {}
        if tlt_data:
            sorted_by_acq = sorted(tlt_data.values(), key=lambda r: r['acq_order'])
            running = 0.0
            for r in sorted_by_acq:
                cum_dose_by_acq[r['acq_order']] = running
                running += r['dose_e_per_A2']

        _MDOC_KEYS = ('sub_frame_path', 'mdoc_defocus', 'target_defocus',
                      'datetime', 'stage_x', 'stage_y', 'stage_z',
                      'exposure_time', 'num_subframes')

        for f in data['frames']:
            f['overlap_pct']  = compute_overlap(f['tx'], f['ty'], W, H)
            f['is_reference'] = (f['tx'] == 0.0 and f['ty'] == 0.0)
            f['is_flagged']   = (f['overlap_pct'] < args.threshold
                                 and not f['is_reference'])
            ctf = ctf_data.get(f['sec'], {})
            f['defocus1_A']      = ctf.get('defocus1_A')
            f['defocus2_A']      = ctf.get('defocus2_A')
            f['mean_defocus_A']  = ctf.get('mean_defocus_A')
            f['mean_defocus_um'] = ctf.get('mean_defocus_um')
            f['astig_A']         = ctf.get('astig_A')
            f['astig_um']        = ctf.get('astig_um')
            f['astig_angle_deg'] = ctf.get('astig_angle_deg')
            f['cc']              = ctf.get('cc')
            f['fit_spacing_A']   = ctf.get('fit_spacing_A')

            # _TLT.txt enrichment (row N = SEC N, 1-indexed)
            tlt = tlt_data.get(f['sec'], {})
            f['nominal_tilt']             = tlt.get('nominal_tilt')
            f['acq_order']                = tlt.get('acq_order')
            f['dose_e_per_A2']            = tlt.get('dose_e_per_A2')
            f['z_value']                  = tlt.get('z_value')
            f['cumulative_dose_e_per_A2'] = (
                round(cum_dose_by_acq[tlt['acq_order']], 2)
                if tlt.get('acq_order') is not None else None
            )

            # mdoc enrichment (keyed by z_value = acq_order - 1)
            mdoc = mdoc_data.get(f['z_value'], {}) if f['z_value'] is not None else {}
            for k in _MDOC_KEYS:
                f[k] = mdoc.get(k)

        # Enrich dark frames: DarkFrame col 2 (frame_b) is the 1-indexed row
        # number in _TLT.txt for that dark frame — direct lookup, no tilt matching.
        for df in data['dark_frames']:
            tlt = tlt_data.get(df['frame_b'], {})
            if tlt:
                df['nominal_tilt']             = tlt['nominal_tilt']
                df['acq_order']                = tlt['acq_order']
                df['dose_e_per_A2']            = tlt['dose_e_per_A2']
                df['z_value']                  = tlt['z_value']
                df['cumulative_dose_e_per_A2'] = round(
                    cum_dose_by_acq.get(tlt['acq_order'], 0.0), 2)
                mdoc = mdoc_data.get(tlt['z_value'], {})
                for k in _MDOC_KEYS:
                    df[k] = mdoc.get(k)
            else:
                for k in ('nominal_tilt', 'acq_order', 'dose_e_per_A2',
                          'z_value', 'cumulative_dose_e_per_A2', *_MDOC_KEYS):
                    df.setdefault(k, None)

        all_ts[ts_name] = data

    # ── Compute global axis ranges ────────────────────────────────────────────
    all_tilts, all_defocus, all_spacing = [], [], []
    for data in all_ts.values():
        for f in data['frames']:
            all_tilts.append(f['tilt'])
            if f.get('mean_defocus_um') is not None:
                all_defocus.append(f['mean_defocus_um'])
            if f.get('fit_spacing_A') is not None:
                all_spacing.append(f['fit_spacing_A'])

    tilt_min = min(all_tilts) - 5
    tilt_max = max(all_tilts) + 5
    abs_tilt_max = max(abs(min(all_tilts)), abs(max(all_tilts)))

    if all_defocus:
        defocus_min = max(0.0, float(np.percentile(all_defocus,  2))) - 0.2
        defocus_max = float(np.percentile(all_defocus, 98)) + 0.2
    else:
        defocus_min, defocus_max = 0.0, 6.0

    if all_spacing:
        fit_spacing_min = float(np.percentile(all_spacing,  5))
        fit_spacing_max = float(np.percentile(all_spacing, 95))
    else:
        fit_spacing_min, fit_spacing_max = 8.0, 30.0

    global_ranges = {
        'tilt_min':        tilt_min,
        'tilt_max':        tilt_max,
        'abs_tilt_max':    abs_tilt_max,
        'defocus_min':     defocus_min,
        'defocus_max':     defocus_max,
        'fit_spacing_min': fit_spacing_min,
        'fit_spacing_max': fit_spacing_max,
    }

    print(f'Global tilt range   : {tilt_min:.1f}° → {tilt_max:.1f}°')
    print(f'Global defocus range: {defocus_min:.2f} → {defocus_max:.2f} µm  '
          f'(2nd–98th percentile)')
    print(f'Global fit spacing  : {fit_spacing_min:.1f} → {fit_spacing_max:.1f} Å  '
          f'(5th–95th percentile)\n')

    # ── Pass 2: generate console output, JSON, TSV, plots ────────────────────
    sep           = '─' * 70
    all_parsed    = {}
    flagged_rows  = []
    ts_entries    = []
    total_flagged = 0

    for ts_name, data in all_ts.items():
        bad_frames    = [f for f in data['frames'] if f['is_flagged']]
        n_bad         = len(bad_frames)
        total_flagged += n_bad

        # Console output
        status = '✗' if n_bad else '✓'
        print(sep)
        print(f'  {status}  {ts_name}   {n_bad} frame(s) below {args.threshold}%  '
              f'| {len(data["dark_frames"])} dark frame(s)')
        if bad_frames:
            print(f'     {"SEC":>4}  {"Tilt (°)":>9}  {"TX (px)":>10}  '
                  f'{"TY (px)":>10}  {"Overlap":>8}')
            for f in bad_frames:
                print(f'     {f["sec"]:>4}  {f["tilt"]:>9.2f}  {f["tx"]:>10.1f}  '
                      f'{f["ty"]:>10.1f}  {f["overlap_pct"]:>7.1f}%')
                flagged_rows.append({
                    'ts':          ts_name,
                    'sec':         f['sec'],
                    'tilt':        f['tilt'],
                    'tx':          f['tx'],
                    'ty':          f['ty'],
                    'overlap_pct': f['overlap_pct'],
                })

        # Store for JSON
        all_parsed[ts_name] = {
            'file':         str(in_dir / f'{ts_name}.aln'),
            'width':        data['width'],
            'height':       data['height'],
            'total_frames': data['total_frames'],
            'alpha_offset': data['alpha_offset'],
            'beta_offset':  data['beta_offset'],
            'thickness':    data['thickness'],
            'thickness_nm': data['thickness_nm'],
            'angpix':       data['angpix'],
            'num_patches':  data['num_patches'],
            'dark_frames':  data['dark_frames'],
            'frames':       data['frames'],
        }

        # Plot
        png_name = f'{ts_name}.png'
        plot_tilt_series(ts_name, data, args.threshold,
                         str(out_dir / png_name), global_ranges)

        ts_entries.append({
            'name':     ts_name,
            'png':      png_name,
            'n_bad':    n_bad,
            'n_frames': len(data['frames']),
            'n_dark':   len(data['dark_frames']),
        })

    print(sep)

    # ── Global summary plot ───────────────────────────────────────────────────
    summary_png = 'global_summary.png'
    plot_global_summary(all_ts, args.threshold, global_ranges,
                        str(out_dir / summary_png))
    ts_entries.insert(0, {
        'name':     '[Summary] Global Analysis',
        'png':      summary_png,
        'n_bad':    0,
        'n_frames': sum(len(d['frames']) for d in all_ts.values()),
        'n_dark':   0,
    })

    # JSON
    json_path = out_dir / 'alignment_data.json'
    with open(json_path, 'w') as fh:
        json.dump(all_parsed, fh, indent=2)

    # Flagged TSV
    tsv_path = out_dir / 'flagged_frames.tsv'
    with open(tsv_path, 'w') as fh:
        fh.write('ts_name\tsec\ttilt\ttx\tty\toverlap_pct\n')
        for row in flagged_rows:
            fh.write(
                f'{row["ts"]}\t{row["sec"]}\t{row["tilt"]:.2f}\t'
                f'{row["tx"]:.3f}\t{row["ty"]:.3f}\t{row["overlap_pct"]:.2f}\n'
            )

    # HTML
    html_path = out_dir / 'index.html'
    make_html(ts_entries, str(html_path), args.threshold)

    # Summary
    n_ts_bad = sum(1 for e in ts_entries if e['n_bad'] > 0)
    print(f'\nSummary')
    print(f'  Tilt series processed : {len(ts_entries)}')
    print(f'  TS with flagged frames: {n_ts_bad}')
    print(f'  Total flagged frames  : {total_flagged}')
    print(f'\nOutput')
    print(f'  Plots          : {out_dir}/<ts-name>.png')
    print(f'  HTML viewer    : {html_path}')
    print(f'  Alignment JSON : {json_path}')
    print(f'  Flagged TSV    : {tsv_path}')


if __name__ == '__main__':
    main()
