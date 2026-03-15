"""
analyse subcommand — parse AreTomo .aln files, produce per-TS plots,
a global summary PNG, an HTML viewer, alignment JSON, and flagged TSV.
"""

import sys
import csv
import time
import json
import shutil
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it  # type: ignore[assignment]



from aretomo3_preprocess.shared.project_json import (
    load as _load_project,
    update_section, update_section_once, args_to_dict,
)
from aretomo3_preprocess.shared.project_state import (
    get_angpix, get_tlt_dir, get_gain_check_dir,
)
from aretomo3_preprocess.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file,
    _float_or_none,
)
from aretomo3_preprocess.shared.geometry import compute_overlap, rotated_rect_corners
from aretomo3_preprocess.shared.colours import (
    OVL_CMAP, OVL_NORM, RES_CMAP,
    _ovl_colour, _ovl_sm, _res_sm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tilt-series plot  (2 × 2 panels)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tilt_series(ts_name, data, threshold, out_path, global_ranges,
                     prev_data=None, vol_path=None):
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
    tilt_xlim    = (global_ranges['tilt_min'],    global_ranges['tilt_max'])
    defocus_ylim = (global_ranges['defocus_min'], global_ranges['defocus_max'])

    # ── Try to load tomogram volume for projections ──────────────────────────
    vol_projs = None   # None if unavailable, else dict with xy, vox, slab_a
    if vol_path is not None and Path(vol_path).exists():
        try:
            import mrcfile as _mrcfile
            # Use mmap so only the needed Z slab is read from disk, not the full volume
            with _mrcfile.mmap(vol_path, mode='r', permissive=True) as mrc:
                vox = float(mrc.voxel_size.x) or 1.0
                nz, ny, nx = mrc.data.shape
                hp = max(1, int(round(150.0 / vox)))   # half-slab in px
                zc = nz // 2
                zs, ze = max(0, zc - hp), min(nz, zc + hp)
                slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)
            vol_projs = {
                'xy':     slab.mean(axis=0),            # (ny, nx)
                'vox':    vox,
                'slab_a': (ze - zs) * vox,
            }
        except Exception:
            pass

    # ── Create figure layout ─────────────────────────────────────────────────
    if vol_projs is not None:
        # Left: XY projection  |  Right: 2×2 analysis panels
        fig   = plt.figure(figsize=(24, 12))
        outer = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.35)
        right = outer[1].subgridspec(2, 2, hspace=0.42, wspace=0.38)
        ax_xy = fig.add_subplot(outer[0])
        ax1   = fig.add_subplot(right[0, 0])   # overlap
        ax2   = fig.add_subplot(right[0, 1])   # frame position
        ax4   = fig.add_subplot(right[1, 0])   # defocus      (lower-left)
        ax3   = fig.add_subplot(right[1, 1])   # tilt coverage (lower-right)
    else:
        fig = plt.figure(figsize=(20, 12))
        gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])   # overlap
        ax2 = fig.add_subplot(gs[0, 1])   # frame position
        ax4 = fig.add_subplot(gs[1, 0])   # defocus      (lower-left)
        ax3 = fig.add_subplot(gs[1, 1])   # tilt coverage (lower-right)

    fig.suptitle(
        f'{ts_name}   |   {len(frames)} aligned frames   |   '
        f'{len(dark_frames)} dark frames   |   '
        f'{n_bad} frame(s) below {threshold}% overlap threshold',
        fontsize=12, fontweight='bold',
    )

    # ── Panel 1 : Overlap % vs tilt angle ────────────────────────────────────
    # Corrected tilt = nominal + alpha_offset.  When comparing across runs the
    # alpha_offset may differ, so we re-express the previous tilts in the
    # current run's alpha space:  prev_tilt - prev_alpha + current_alpha
    curr_alpha = data.get('alpha_offset') or 0.0
    if prev_data is not None:
        prev_alpha  = prev_data.get('alpha_offset') or 0.0
        prev_frames = prev_data.get('frames', [])
        if prev_frames:
            alpha_delta   = curr_alpha - prev_alpha
            prev_tilts    = [f['tilt'] + alpha_delta for f in prev_frames]
            prev_overlaps = [f['overlap_pct']        for f in prev_frames]
            ax1.scatter(prev_tilts, prev_overlaps, color='#aaaaaa', s=25,
                        marker='o', alpha=0.5, zorder=2, label='previous run')

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
    ax3.axhline(0, color='#555555', lw=0.8, zorder=1)
    ax3.axvline(0, color='#888888', lw=0.6, ls=':', zorder=1)

    # Dark frames: dashed grey lines through origin
    for dt in dark_tilts:
        tr   = np.radians(dt)
        ctr  = np.cos(tr)
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
        ax3.scatter(ctr, str_, color=ovl_cols[i],
                    s=sz, marker=mk, edgecolors=ec, linewidths=0.8, zorder=4)

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

    # ── XY tomogram projection (central 300 Å slab) ─────────────────────────
    if vol_projs is not None:
        img  = vol_projs['xy']
        p1   = np.percentile(img, 1)
        p99  = np.percentile(img, 99)
        ax_xy.imshow(img, cmap='gray', vmin=p1, vmax=p99,
                     aspect='equal', origin='lower', interpolation='nearest')
        slab = vol_projs['slab_a']
        vox  = vol_projs['vox']
        ax_xy.set_title(f'XY projection\ncentral {slab:.0f} Å slab  ({vox:.1f} Å/px)',
                        fontsize=9)
        ax_xy.set_xlabel('X (px)', fontsize=8)
        ax_xy.set_ylabel('Y (px)', fontsize=8)
        ax_xy.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity / consistency checks
# ─────────────────────────────────────────────────────────────────────────────

def _validate_ts(data, tlt_data, mdoc_data, mrc_path=None):
    """
    Run consistency checks across the parsed data sources for one tilt series.

    Checks performed:
      1. MRC header nx/ny/nz vs .aln width/height/total_frames
      2. _TLT.txt nominal_tilt + AlphaOffset ≈ .aln corrected tilt  (per frame)
      3. Every aligned frame's z_value maps to a key in the mdoc
      4. _TLT.txt rows are fully covered by aligned SECs ∪ dark frame_bs
      5. Mdoc TiltAngle ≈ _TLT.txt nominal_tilt  (via z_value linkage)

    Returns a list of warning strings (empty = all OK).
    """
    warnings = []
    alpha = data.get('alpha_offset') or 0.0

    # 1. MRC header
    if mrc_path is not None and mrc_path.exists():
        try:
            import mrcfile
            with mrcfile.mmap(mrc_path, mode='r', permissive=True) as m:
                nx = int(m.header.nx)
                ny = int(m.header.ny)
                nz = int(m.header.nz)
            if nx != data['width']:
                warnings.append(f'MRC nx={nx} ≠ .aln width={data["width"]}')
            if ny != data['height']:
                warnings.append(f'MRC ny={ny} ≠ .aln height={data["height"]}')
            if nz != data['total_frames']:
                warnings.append(
                    f'MRC nz={nz} ≠ .aln total_frames={data["total_frames"]}')
        except Exception as e:
            warnings.append(f'MRC header read failed: {e}')

    # 2. Corrected tilt: TLT nominal + alpha ≈ .aln TILT
    if tlt_data:
        bad = []
        for f in data['frames']:
            tlt = tlt_data.get(f['sec'])
            if tlt is None:
                continue
            if abs(tlt['nominal_tilt'] + alpha - f['tilt']) > 0.05:
                bad.append(f['sec'])
        if bad:
            warnings.append(
                f'{len(bad)} frame(s) |TLT nominal+α − .aln tilt| > 0.05°: '
                f'SEC {bad}')

    # 3. ZValue linkage: every frame's z_value must appear in mdoc
    if mdoc_data:
        bad = [f['sec'] for f in data['frames']
               if f.get('z_value') is not None and f['z_value'] not in mdoc_data]
        if bad:
            warnings.append(
                f'{len(bad)} frame(s) with z_value not found in mdoc: SEC {bad}')

    # 4. TLT row coverage: aligned SECs ∪ dark frame_bs == all TLT rows
    if tlt_data:
        aligned_secs  = {f['sec']      for f in data['frames']}
        dark_frame_bs = {df['frame_b'] for df in data['dark_frames']}
        accounted     = aligned_secs | dark_frame_bs
        tlt_keys      = set(tlt_data.keys())
        missing       = tlt_keys  - accounted
        extra         = accounted - tlt_keys
        if missing:
            warnings.append(
                f'TLT rows with no matching SEC or dark frame_b: {sorted(missing)}')
        if extra:
            warnings.append(
                f'SECs/dark frame_bs with no matching TLT row: {sorted(extra)}')

    # 5. Mdoc TiltAngle ≈ TLT nominal_tilt
    if tlt_data and mdoc_data:
        bad = []
        for f in data['frames']:
            tlt  = tlt_data.get(f['sec'])
            z    = f.get('z_value')
            if tlt is None or z is None:
                continue
            mdoc_tilt = mdoc_data.get(z, {}).get('tilt_angle')
            if mdoc_tilt is None:
                continue
            if abs(mdoc_tilt - tlt['nominal_tilt']) > 0.05:
                bad.append(f['sec'])
        if bad:
            warnings.append(
                f'{len(bad)} frame(s) |mdoc TiltAngle − TLT nominal_tilt| > 0.05°: '
                f'SEC {bad}')

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Global summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_summary(all_ts, threshold, global_ranges, out_path, prev_ts=None):
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
      (2,1) Lamella thickness AlignZ (px) per TS
      (2,2) Flagged frames per TS
    """
    n_ts = len(all_ts)

    # Per-TS accumulators
    n_total_list      = []
    n_aligned_list    = []
    n_passing_list    = []
    n_flagged_list    = []
    rot_list          = []
    alpha_offset_list = []
    thickness_list    = []
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

        if data.get('thickness') is not None:
            thickness_list.append(data['thickness'])
            if thickness_angpix is None and data.get('angpix') is not None:
                thickness_angpix = data['angpix']

        for f in frames:
            overlap_all.append(f['overlap_pct'])
            if f.get('mean_defocus_um') is not None:
                defocus_all.append(f['mean_defocus_um'])
            if f.get('fit_spacing_A') is not None:
                spacing_all.append(f['fit_spacing_A'])
            if f.get('astig_um') is not None:
                astig_all.append(f['astig_um'])

    def _median_vline(ax, data, color='steelblue'):
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
        # Clip x-axis to 2nd–98th percentile so extreme outliers don't
        # compress the histogram into an uninformative spike.
        p2  = float(np.percentile(spacing_all,  2))
        p98 = float(np.percentile(spacing_all, 98))
        clipped = [v for v in spacing_all if p2 <= v <= p98]
        if prev_ts is not None:
            prev_spacing = [f['fit_spacing_A'] for d in prev_ts.values()
                            for f in d['frames'] if f.get('fit_spacing_A') is not None]
            if prev_spacing:
                prev_clipped = [v for v in prev_spacing if p2 <= v <= p98]
                ax.hist(prev_clipped, bins=50, color='#aaaaaa', alpha=0.5,
                        edgecolor='none', label='previous run', zorder=1)
        ax.hist(clipped, bins=50, color='darkorange',
                edgecolor='white', linewidth=0.3, label='current run', zorder=2)
        _median_vline(ax, spacing_all)   # median of full data for accuracy
        ax.set_xlim(p2, p98)
        ax.set_xlabel('Fit spacing (Å)')
        ax.set_ylabel('# Frames')
        ax.set_title('Estimated Resolution (fit_spacing_A)\n'
                     '(x-axis clipped to 2nd–98th percentile)')
        ax.legend(fontsize=8)
    else:
        _no_data(ax)

    # ── (1,2) % Overlap ───────────────────────────────────────────────────
    ax = axes[1, 2]
    if overlap_all:
        if prev_ts is not None:
            prev_ovl = [f['overlap_pct'] for d in prev_ts.values() for f in d['frames']]
            if prev_ovl:
                ax.hist(prev_ovl, bins=50, color='#aaaaaa', alpha=0.5,
                        edgecolor='none', label='previous run', zorder=1)
        ax.hist(overlap_all, bins=50, color='mediumseagreen',
                edgecolor='white', linewidth=0.3, label='current run', zorder=2)
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

    # ── (2,1) Lamella Thickness AlignZ ────────────────────────────────────
    ax = axes[2, 1]
    if thickness_list:
        ax.hist(thickness_list, bins=30, color='slateblue',
                edgecolor='white', linewidth=0.3)
        _median_vline(ax, thickness_list)
        ax.set_xlabel('Lamella Thickness — AlignZ (px)')
        ax.set_ylabel('# Tilt series')
        if thickness_angpix is not None:
            title = f'Lamella Thickness — AlignZ (px)\n(angpix = {thickness_angpix} Å/px from run)'
        else:
            title = 'Lamella Thickness — AlignZ (px)'
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

def make_html(ts_entries, out_path, threshold, gain_check=None, selection=None,
              ratings=None):
    """
    Generate the HTML report.

    Parameters
    ----------
    ts_entries  : list of dicts  (name, png, n_bad, n_frames, n_dark)
    out_path    : str            path to write index.html
    threshold   : float          overlap threshold (%)
    gain_check  : dict or None   gain_check section from aretomo3_project.json;
                                 when provided a 'Gain Transform Check' tab is
                                 added before the tilt-series viewer.
    selection   : dict or None   {ts_name: bool} from select-ts; when provided
                                 a 'Selected only' toggle is shown in the viewer.
    ratings     : dict or None   {ts_name: int} loaded from ts_ratings.csv;
                                 pre-populates star ratings on page load
                                 (localStorage overrides for any TS the user
                                 re-rates interactively).
    """
    # Determine per-entry selected flag (summary/lamella entries are always selected)
    def _is_selected(e):
        if selection is None:
            return True
        if e['name'].startswith('['):
            return True
        return selection.get(e['name'], True)

    selected_flags = [1 if _is_selected(e) else 0 for e in ts_entries]
    has_selection  = selection is not None
    n_sel = sum(selected_flags)

    options_html = '\n'.join(
        f'    <option value="{i}" data-bad="{e["n_bad"]}" data-selected="{selected_flags[i]}">'
        f'{"[!] " if e["n_bad"] > 0 else "      "}'
        f'{"" if selected_flags[i] else "[x] "}'
        f'{e["name"]}  ({e["n_bad"]} flagged / {e["n_frames"]} frames)'
        f'</option>'
        for i, e in enumerate(ts_entries)
    )
    images_js      = json.dumps([e['png'] for e in ts_entries])
    titles_js      = json.dumps([
        f'{e["name"]}  —  {e["n_frames"]} aligned frames  |  '
        f'{e["n_dark"]} dark frames  |  {e["n_bad"]} below {threshold}% threshold'
        for e in ts_entries
    ])
    selected_js    = json.dumps(selected_flags)
    n              = len(ts_entries)

    # ── Gain-check tab content (only when gain_check is provided) ─────────────
    has_gain = gain_check is not None
    if has_gain:
        best  = gain_check.get('best_transform', 'unknown')
        rg    = gain_check.get('aretomo3_rot_gain',  '?')
        fg    = gain_check.get('aretomo3_flip_gain', '?')
        flags = f'-RotGain {rg} -FlipGain {fg}'
        n_mov = gain_check.get('n_movies_tested', '?')
        acq   = gain_check.get('acq_range', ['?', '?'])
        tilt  = gain_check.get('tilt_range_deg', ['?', '?'])
        ts_gc = gain_check.get('timestamp', '')[:10]

        rows_html = ''
        for name, s in gain_check.get('scores', {}).items():
            marker   = '  \u2190 best' if name == best else ''
            style    = 'color:#66bb6a;font-weight:bold' if name == best else ''
            cv_mul_s = f"{s['cv_mul']:.4f}" if s.get('cv_mul') is not None else 'n/a'
            cv_div_s = f"{s['cv_div']:.4f}" if s.get('cv_div') is not None else 'n/a'
            rows_html += (
                f'<tr style="{style}"><td>{name}{marker}</td>'
                f'<td>{cv_mul_s}</td><td>{cv_div_s}</td></tr>\n'
            )

        gain_tab_btn  = '<button class="tab-btn active" onclick="switchTab(\'gain\')">Gain Transform Check</button>'
        ts_tab_btn    = '<button class="tab-btn" onclick="switchTab(\'ts\')">Tilt Series Analysis</button>'
        gain_section  = f"""
  <div id="tab-gain" class="tab-section">
    <div class="gc-card">
      <div class="gc-best">Best transform: {best}</div>
      <div class="gc-flags">{flags}</div>
      <table class="gc-table">
        <tr><th>Transform</th><th>CV &#x2193; raw&#xd7;gain</th><th>CV &#x2193; raw&#xf7;gain</th></tr>
        {rows_html}
      </table>
      <div class="gc-meta">
        Acq &le; {gain_check.get("acq_order_threshold", "?")} filter:
        {gain_check.get("n_movies_after_filter", "?")} movies &nbsp;|&nbsp;
        Sampled: {n_mov} &nbsp;|&nbsp;
        Acq range: {acq[0]:03d}&#x2013;{acq[1]:03d} &nbsp;|&nbsp;
        Tilt: {tilt[0]:+.2f}&#xb0; &#x2192; {tilt[1]:+.2f}&#xb0; &nbsp;|&nbsp;
        {ts_gc}
      </div>
    </div>
    <div class="gc-imgs">
      <img src="corrected_averages.png" alt="Corrected averages per transform">
      <img src="cv_vs_nmovies.png" alt="CV convergence">
    </div>
  </div>"""
        ts_section_style = 'display:none'
        tab_bar          = f'<div id="tab-bar">{gain_tab_btn}{ts_tab_btn}</div>'
        tab_init_js      = "let activeTab = 'gain';"
    else:
        gain_section     = ''
        ts_section_style = ''
        tab_bar          = ''
        tab_init_js      = "let activeTab = 'ts';"

    # ── Tab switching JS (only needed when there are two tabs) ────────────────
    tab_switch_js = """
    function switchTab(name) {
      activeTab = name;
      document.querySelectorAll('.tab-section').forEach(s => {
        s.style.display = (s.id === 'tab-' + name) ? '' : 'none';
      });
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b.textContent.toLowerCase().includes(name));
      });
    }
    """ if has_gain else ''

    # ts_name for each entry (None for summary/lamella overview entries)
    ts_names_for_rating = [
        None if e['name'].startswith('[') else e['name']
        for e in ts_entries
    ]
    ts_names_js    = json.dumps(ts_names_for_rating)
    embedded_ratings_js = json.dumps(ratings or {})
    session_key    = Path(out_path).parent.name or 'aretomo'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AreTomo3 Analysis Report</title>
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
    h1 {{ margin-bottom: 14px; font-size: 1.25em; color: #90caf9; letter-spacing: 0.03em; }}

    /* ── Tab bar ── */
    #tab-bar {{
      display: flex; gap: 6px; margin-bottom: 20px;
    }}
    .tab-btn {{
      padding: 8px 22px; font-size: 0.95em; border: none; border-radius: 6px;
      background: #1e2a45; color: #90a4ae; cursor: pointer; transition: all 0.2s;
    }}
    .tab-btn.active {{ background: #1565c0; color: white; }}
    .tab-btn:hover:not(.active) {{ background: #2a3f60; color: #e0e0e0; }}

    /* ── Gain check tab ── */
    .tab-section {{ width: 100%; max-width: 1380px; }}
    .gc-card {{
      background: #1e2a45; border-radius: 10px; padding: 20px 24px;
      margin-bottom: 20px; max-width: 680px;
    }}
    .gc-best {{ color: #66bb6a; font-size: 1.2em; font-weight: bold; }}
    .gc-flags {{
      font-family: monospace; font-size: 1em; color: #ffcc80;
      background: #0d1b2a; padding: 5px 12px; border-radius: 6px;
      display: inline-block; margin-top: 10px;
    }}
    .gc-table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
    .gc-table th, .gc-table td {{
      padding: 7px 14px; text-align: left; border-bottom: 1px solid #2e3f5c;
    }}
    .gc-table th {{ color: #90caf9; font-weight: normal; }}
    .gc-meta {{ color: #78909c; font-size: 0.82em; margin-top: 12px; }}
    .gc-imgs {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .gc-imgs img {{
      max-width: 720px; width: 100%; border-radius: 8px; border: 1px solid #2e3f5c;
    }}

    /* ── TS viewer tab ── */
    #controls {{
      display: flex; align-items: center; gap: 10px;
      margin-bottom: 10px; flex-wrap: wrap; justify-content: center;
    }}
    button.nav-btn {{
      padding: 8px 20px; font-size: 1em; border: none; border-radius: 6px;
      background: #1565c0; color: white; cursor: pointer; transition: background 0.2s;
    }}
    button.nav-btn:hover {{ background: #0d47a1; }}
    select {{
      padding: 7px 10px; font-size: 0.88em; border-radius: 6px;
      background: #1e2a45; color: #e0e0e0; border: 1px solid #445;
      max-width: 380px; cursor: pointer;
    }}
    #counter {{ font-size: 0.88em; color: #90a4ae; min-width: 60px; text-align: center; }}
    #title {{
      font-size: 0.92em; color: #ffcc80; text-align: center;
      margin-bottom: 10px; min-height: 1.4em;
    }}
    #img-wrap {{
      width: 100%; max-width: 1380px; background: #0d1b2a;
      border-radius: 10px; padding: 12px;
      display: flex; align-items: center; justify-content: center;
    }}
    #main-img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    #hint {{ font-size: 0.75em; color: #546e7a; margin-top: 12px; }}
    #progress {{
      width: 100%; max-width: 1380px; height: 4px;
      background: #1e2a45; border-radius: 2px; margin-bottom: 10px;
    }}
    #progress-bar {{
      height: 100%; background: #1565c0; border-radius: 2px; transition: width 0.2s;
    }}

    /* ── Star ratings ── */
    #rating-bar {{
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 8px; width: 100%; max-width: 1380px;
    }}
    .star {{
      font-size: 1.9em; cursor: pointer; color: #37474f;
      transition: color 0.10s; user-select: none; line-height: 1;
    }}
    .star.on {{ color: #ffc107; }}
    #rating-label {{ font-size: 0.84em; color: #78909c; min-width: 82px; }}
    #btn-export {{ margin-left: auto; padding: 6px 18px; font-size: 0.84em; }}
  </style>
</head>
<body>
  <h1>AreTomo3 Analysis Report</h1>

  {tab_bar}
{gain_section}
  <div id="tab-ts" class="tab-section" style="{ts_section_style}">
    <div id="controls">
      <button class="nav-btn" id="btn-prev">&#8592; Prev</button>
      <select id="ts-select">
{options_html}
      </select>
      <button class="nav-btn" id="btn-next">Next &#8594;</button>
      <span id="counter">1&nbsp;/&nbsp;{n}</span>
      <button class="nav-btn" id="btn-filter"
              style="font-size:0.82em;background:#37474f;{'display:none' if not has_selection else ''}">
        Selected only ({n_sel})
      </button>
      <button class="nav-btn" id="btn-reload-sel"
              style="font-size:0.82em;background:#37474f;"
              title="Re-fetch ts-select.csv from the same directory as this HTML (requires HTTP server)">
        &#8635; Reload selection
      </button>
      <input type="file" id="file-sel-input" accept=".csv"
             style="display:none">
      <button class="nav-btn" id="btn-load-csv"
              style="font-size:0.82em;background:#37474f;"
              title="Load any ts-select.csv from your computer (works with file://)">
        &#128193; Load CSV&#8230;
      </button>
    </div>

    <div id="progress"><div id="progress-bar"></div></div>
    <div id="title"></div>

    <div id="rating-bar">
      <span id="stars">
        <span class="star" data-val="1">&#9733;</span>
        <span class="star" data-val="2">&#9733;</span>
        <span class="star" data-val="3">&#9733;</span>
        <span class="star" data-val="4">&#9733;</span>
        <span class="star" data-val="5">&#9733;</span>
      </span>
      <span id="rating-label">&#8212;</span>
      <button class="nav-btn" id="btn-export">Export ratings CSV</button>
    </div>

    <div id="img-wrap">
      <img id="main-img" src="" alt="tilt series plot">
    </div>

    <p id="hint">Keyboard: &#8592; &#8594; to navigate between tilt series</p>
  </div>

  <script>
    {tab_init_js}
    {tab_switch_js}

    const images        = {images_js};
    const titles        = {titles_js};
    const selectedFlags = {selected_js};
    const n             = images.length;
    let   idx           = 0;
    let   showSelOnly   = false;
    let   visIndices    = Array.from({{length: n}}, (_, i) => i);

    const img     = document.getElementById('main-img');
    const sel     = document.getElementById('ts-select');
    const ctr     = document.getElementById('counter');
    const ttl     = document.getElementById('title');
    const pbar    = document.getElementById('progress-bar');
    const btnFilt = document.getElementById('btn-filter');

    function rebuildVis() {{
      visIndices = [];
      for (let i = 0; i < n; i++) {{
        const opt = sel.options[i];
        if (showSelOnly && selectedFlags[i] === 0) {{
          opt.style.display = 'none';
        }} else {{
          opt.style.display = '';
          visIndices.push(i);
        }}
        opt.style.color = (selectedFlags[i] === 0) ? '#546e7a' : '';
      }}
    }}

    function show(i) {{
      idx = ((i % n) + n) % n;
      img.src          = images[idx];
      ttl.textContent  = titles[idx];
      ctr.innerHTML    = (visIndices.indexOf(idx) + 1) + '&nbsp;/&nbsp;' + visIndices.length;
      sel.value        = idx;
      pbar.style.width = ((visIndices.indexOf(idx) + 1) / visIndices.length * 100).toFixed(1) + '%';
      showRating(idx);
    }}

    function showStep(step) {{
      const pos    = visIndices.indexOf(idx);
      const newPos = ((pos + step) % visIndices.length + visIndices.length) % visIndices.length;
      show(visIndices[newPos]);
    }}

    if (btnFilt) {{
      btnFilt.addEventListener('click', () => {{
        showSelOnly = !showSelOnly;
        btnFilt.style.background = showSelOnly ? '#2e7d32' : '#37474f';
        rebuildVis();
        if (!visIndices.includes(idx)) show(visIndices[0] || 0);
        else show(idx);
      }});
    }}

    document.getElementById('btn-prev').onclick = () => showStep(-1);
    document.getElementById('btn-next').onclick = () => showStep(1);
    sel.onchange = () => show(parseInt(sel.value));
    document.addEventListener('keydown', e => {{
      if (e.key === 'ArrowLeft')  showStep(-1);
      if (e.key === 'ArrowRight') showStep(1);
    }});
    rebuildVis();

    // ── Star ratings ──────────────────────────────────────────────────────
    const tsNames        = {ts_names_js};
    const storageKey     = 'aretomo_ratings_{session_key}';
    const embeddedRatings = {embedded_ratings_js};
    // Merge: embedded CSV ratings are the baseline; localStorage overrides
    // any TS the user has re-rated in this browser session.
    let   ratings    = Object.assign({{}}, embeddedRatings,
                           JSON.parse(localStorage.getItem(storageKey) || '{{}}'));

    const stars     = document.querySelectorAll('.star');
    const ratingLbl = document.getElementById('rating-label');
    const ratingBar = document.getElementById('rating-bar');
    const origOpts  = Array.from(sel.options).map(o => o.textContent);
    const starChars = ['', '\u2605', '\u2605\u2605', '\u2605\u2605\u2605',
                       '\u2605\u2605\u2605\u2605', '\u2605\u2605\u2605\u2605\u2605'];

    function showRating(i) {{
      const name = tsNames[i];
      const val  = name ? (ratings[name] || 0) : 0;
      stars.forEach(s => s.classList.toggle('on', parseInt(s.dataset.val) <= val));
      ratingBar.style.opacity = name ? '1' : '0.3';
      ratingLbl.textContent   = !name ? '\u2014'
                              : val === 0 ? 'Not rated' : starChars[val] + ' (' + val + '/5)';
    }}

    function saveRating(i, val) {{
      const name = tsNames[i];
      if (!name) return;
      if (ratings[name] === val) {{ delete ratings[name]; }}
      else {{ ratings[name] = val; }}
      localStorage.setItem(storageKey, JSON.stringify(ratings));
      showRating(i);
      const r = ratings[name] || 0;
      sel.options[i].textContent = origOpts[i] + (r > 0 ? '  ' + starChars[r] : '');
    }}

    stars.forEach(s => {{
      s.addEventListener('click', () => saveRating(idx, parseInt(s.dataset.val)));
      s.addEventListener('mouseover', () => {{
        const v = parseInt(s.dataset.val);
        stars.forEach(s2 => s2.style.color = parseInt(s2.dataset.val) <= v ? '#ffdc6e' : '');
      }});
      s.addEventListener('mouseout', () => {{
        stars.forEach(s2 => {{ s2.style.color = ''; }});
        showRating(idx);
      }});
    }});

    // Restore ratings into dropdown on load
    for (let i = 0; i < n; i++) {{
      const name = tsNames[i];
      if (name && ratings[name]) {{
        sel.options[i].textContent = origOpts[i] + '  ' + starChars[ratings[name]];
      }}
    }}

    document.getElementById('btn-export').addEventListener('click', () => {{
      const rows = ['ts_name,rating'];
      [...new Set(tsNames.filter(Boolean))].sort().forEach(name => {{
        if (ratings[name]) rows.push(name + ',' + ratings[name]);
      }});
      if (rows.length === 1) {{ alert('No ratings to export yet.'); return; }}
      const a   = document.createElement('a');
      a.href    = URL.createObjectURL(
        new Blob([rows.join('\\n') + '\\n'], {{type: 'text/csv'}}));
      a.download = 'ts_ratings.csv';
      a.click();
    }});

    // ── Dynamic CSV loading ────────────────────────────────────────────────
    // Load ts_ratings.csv and ts-select.csv from the same directory as
    // this HTML file at page-load time.  Falls back silently to embedded
    // data if the files are not accessible (e.g. file:// protocol in Chrome).
    function _parseCSV(text) {{
      const lines   = text.trim().split('\\n');
      if (lines.length < 2) return [];
      const headers = lines[0].split(',').map(h => h.trim());
      return lines.slice(1).map(line => {{
        const vals = line.split(',').map(v => v.trim());
        const obj  = {{}};
        headers.forEach((h, i) => {{ obj[h] = (vals[i] !== undefined ? vals[i] : ''); }});
        return obj;
      }});
    }}

    function _applyRatingsCSV(text) {{
      _parseCSV(text).forEach(row => {{
        if (row.ts_name && row.rating) {{
          const r = parseInt(row.rating);
          if (!isNaN(r) && r > 0) ratings[row.ts_name] = r;
        }}
      }});
      for (let i = 0; i < n; i++) {{
        const name = tsNames[i];
        const r    = name ? (ratings[name] || 0) : 0;
        sel.options[i].textContent = origOpts[i] + (r > 0 ? '  ' + starChars[r] : '');
      }}
      showRating(idx);
    }}

    function _applySelectionCSV(text) {{
      const byName = {{}};
      _parseCSV(text).forEach(row => {{
        if (row.ts_name) byName[row.ts_name] = parseInt(row.selected || '0');
      }});
      let changed = false;
      for (let i = 0; i < n; i++) {{
        const name = tsNames[i];
        if (name && name in byName && selectedFlags[i] !== byName[name]) {{
          selectedFlags[i] = byName[name];
          sel.options[i].setAttribute('data-selected', byName[name]);
          changed = true;
        }}
      }}
      if (changed) {{
        rebuildVis();
        if (!visIndices.includes(idx)) show(visIndices[0] || 0);
        else show(idx);
      }}
      // Update 'Selected only' button count
      const btnF = document.getElementById('btn-filter');
      if (btnF) {{
        const nSel = selectedFlags.filter(v => v !== 0).length;
        btnF.style.display = '';
        btnF.textContent   = 'Selected only (' + nSel + ')';
      }}
    }}

    fetch('./ts_ratings.csv')
      .then(r => r.ok ? r.text() : Promise.reject())
      .then(text => _applyRatingsCSV(text))
      .catch(() => {{}});

    fetch('./ts-select.csv')
      .then(r => r.ok ? r.text() : Promise.reject())
      .then(text => _applySelectionCSV(text))
      .catch(() => {{}});

    document.getElementById('btn-reload-sel').addEventListener('click', () => {{
      fetch('./ts-select.csv?_=' + Date.now())
        .then(r => r.ok ? r.text() : Promise.reject())
        .then(text => {{ _applySelectionCSV(text); }})
        .catch(() => {{ alert('ts-select.csv not found in this directory.\nUse the \u201cLoad CSV\u2026\u201d button to browse for the file instead.'); }});
    }});

    // ── File-picker for ts-select.csv (works on file:// protocol) ─────────
    const fileInput = document.getElementById('file-sel-input');
    document.getElementById('btn-load-csv').addEventListener('click', () => {{
      fileInput.value = '';   // allow re-selecting same file
      fileInput.click();
    }});
    fileInput.addEventListener('change', () => {{
      const file = fileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {{ _applySelectionCSV(e.target.result); }};
      reader.readAsText(file);
    }});

    show(0);
  </script>
</body>
</html>
"""
    with open(out_path, 'w') as fh:
        fh.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# Stage position scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def _base_position_label(original_path_str):
    """
    Extract the base Position label from an original mdoc path.
    e.g. '/frames/Position_6_2.mdoc' → 'Position_6'
         '/frames/Position_21_10.mdoc' → 'Position_21'
         '/frames/Position_1.mdoc' → 'Position_1'
    """
    stem = Path(original_path_str).stem   # Position_6_2
    parts = stem.split('_')               # ['Position', '6', '2']
    if len(parts) >= 2:
        return f'{parts[0]}_{parts[1]}'   # Position_6
    return stem


def plot_stage_positions(all_ts, out_dir, n_lamellae=None, lookup=None,
                         cluster_ids_override=None):
    """
    Scatter plots of stage X-Y positions from mdoc data.

    Produces:
      - Overview plot: all lamellae colour-coded, legend with position names
      - Per-lamella plot: scatter (one marker per base Position, labelled with
        ts-XXX, coloured by mean n_tilts) + 2×3 histogram panels showing
        per-lamella statistics with suggested (median) values
      - lamella_positions.csv: ts_name, original_path, base_position, lamella

    Parameters
    ----------
    all_ts      : dict  ts_name → data dict (frames have stage_x, stage_y)
    out_dir     : Path  output directory
    n_lamellae  : int or None   number of K-means clusters; no clustering if None
    lookup      : dict or None  ts_name → original_path (from rename_ts section)

    Returns
    -------
    list of (viewer_label, png_filename) for insertion into the HTML viewer
    """
    from collections import defaultdict
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    ts_names, xs, ys, n_tilts_list = [], [], [], []

    for ts_name, data in all_ts.items():
        sx_vals = [f['stage_x'] for f in data['frames']
                   if f.get('stage_x') is not None]
        sy_vals = [f['stage_y'] for f in data['frames']
                   if f.get('stage_y') is not None]
        if not sx_vals or not sy_vals:
            continue
        ts_names.append(ts_name)
        xs.append(float(np.mean(sx_vals)))
        ys.append(float(np.mean(sy_vals)))
        n_tilts_list.append(len(data['frames']))

    n = len(ts_names)
    if n == 0:
        print('  Stage positions: no stage coordinate data found — skipped')
        return [], {}

    X = np.column_stack([xs, ys])

    # Build base-position labels
    if lookup:
        base_labels = [
            _base_position_label(lookup.get(f'{ts}.mdoc', ts))
            for ts in ts_names
        ]
    else:
        base_labels = ts_names

    original_paths = {ts: lookup.get(f'{ts}.mdoc', '') for ts in ts_names} \
        if lookup else {}

    # K-means clustering (or reuse previous assignments)
    cluster_ids = np.zeros(n, dtype=int)
    n_clusters  = 1
    if cluster_ids_override is not None:
        cluster_ids = np.array(
            [cluster_ids_override.get(ts, 0) for ts in ts_names], dtype=int)
        n_clusters = int(cluster_ids.max()) + 1 if n > 0 else 1
        print(f'  Reused previous lamellae assignments ({n_clusters} lamellae, {n} TS)')
    elif n_lamellae is not None and n_lamellae > 1 and n >= n_lamellae:
        try:
            from sklearn.cluster import KMeans
            import warnings as _warnings
            km = KMeans(n_clusters=n_lamellae, random_state=42, n_init=10)
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                t0 = time.time()
                cluster_ids = km.fit_predict(X)
                t1 = time.time()
            n_clusters = n_lamellae
            print(f'  K-means ({n_lamellae} clusters, {n} TS): {t1 - t0:.1f} s')
        except ImportError:
            print('  WARNING: scikit-learn not installed — K-means clustering skipped')

    cmap = plt.colormaps.get_cmap('tab10')

    # Per-cluster: sorted unique base-position labels (for legend)
    cluster_pos_labels = defaultdict(set)
    for i, lbl in enumerate(base_labels):
        cluster_pos_labels[cluster_ids[i]].add(lbl)
    cluster_pos_labels = {
        c: sorted(v, key=lambda s: int(s.split('_')[1])
                  if len(s.split('_')) > 1 and s.split('_')[1].isdigit() else 0)
        for c, v in cluster_pos_labels.items()
    }

    # ── Write CSV: ts_name, original_path, base_position, lamella ─────────
    csv_path = out_dir / 'lamella_positions.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ts_name', 'original_path', 'base_position', 'lamella'])
        rows = sorted(zip(ts_names, base_labels, cluster_ids),
                      key=lambda r: (int(r[2]), r[0]))
        for ts, base, c in rows:
            writer.writerow([ts, original_paths.get(ts, ''), base, int(c) + 1])
    print(f'  Lamella CSV     : {csv_path}')

    output_entries = []

    # ── Overview plot ──────────────────────────────────────────────────────
    colours = [cmap(c % 10) for c in cluster_ids]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(xs, ys, c=colours, s=60,
               edgecolors='white', linewidths=0.5, zorder=3)

    cluster_str = (f', {n_clusters} lamellae' if n_clusters > 1 else '')
    ax.set_title(f'Lamella position on stage  ({n} tilt series{cluster_str})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Stage X (µm)')
    ax.set_ylabel('Stage Y (µm)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, lw=0.4, alpha=0.4)

    # Annotate one representative ts-XXX per base Position
    seen_base_ov = {}
    for i in sorted(range(n), key=lambda i: ts_names[i]):
        blbl = base_labels[i]
        if blbl not in seen_base_ov:
            seen_base_ov[blbl] = i
    for i in seen_base_ov.values():
        ax.annotate(ts_names[i], xy=(xs[i], ys[i]), xytext=(4, 4),
                    textcoords='offset points', fontsize=5.5,
                    color=cmap(cluster_ids[i] % 10), zorder=5)

    handles = []
    for c in range(n_clusters):
        pos_list = [lbl.replace('Position_', 'P')
                    for lbl in cluster_pos_labels.get(c, [])]
        pos_str  = ', '.join(pos_list)
        if len(pos_str) > 55:
            pos_str = pos_str[:52] + '...'
        lbl_text = (f'Lamella {c+1}: {pos_str}' if n_clusters > 1
                    else f'All: {pos_str}')
        handles.append(Patch(facecolor=cmap(c % 10), edgecolor='white',
                             label=lbl_text))
    ax.legend(handles=handles, fontsize=7, loc='best', framealpha=0.85)

    fig.tight_layout()
    overview_png = 'stage_positions.png'
    fig.savefig(out_dir / overview_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Stage overview  : {out_dir / overview_png}')
    output_entries.append(('[Summary] Stage Positions', overview_png))

    # ── Per-lamella focused plots ──────────────────────────────────────────
    if n_clusters <= 1:
        return output_entries, {}

    n_tilts_arr = np.array(n_tilts_list)
    vmin = int(n_tilts_arr.min())
    vmax = int(n_tilts_arr.max())

    def _hist_panel(ax, data, xlabel, color, fmt='.1f', unit='',
                    show_suggested=True):
        """Draw a histogram with a dashed median line.

        If show_suggested is True (default), the axes title shows
        'Suggested: <median>'.  Set False for informational panels
        (n_tilts, n_flagged, resolution) where the median is descriptive
        rather than a recommended input parameter.
        """
        clean = [v for v in data if v is not None and not np.isnan(v)]
        if not clean:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_axis_off()
            return
        med = float(np.median(clean))
        ax.hist(clean, bins=min(12, max(3, len(clean))),
                color=color, edgecolor='white', linewidth=0.3)
        ax.axvline(med, color='steelblue', lw=1.8, ls='--',
                   label=f'median = {med:{fmt}}{unit}')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('# TS', fontsize=8)
        ax.tick_params(labelsize=7)
        if show_suggested:
            ax.set_title(f'Suggested: {med:{fmt}}{unit}',
                         fontsize=8, color='steelblue', fontweight='bold')
        ax.legend(fontsize=7)

    lamella_stats = {}

    for c in range(n_clusters):
        idx = [i for i, ci in enumerate(cluster_ids) if ci == c]
        if not idx:
            continue

        ts_in_cluster = [ts_names[i] for i in idx]

        # ── Deduplicate to first ts-XXX per base Position ─────────────
        seen_base = {}
        for i in sorted(idx, key=lambda i: ts_names[i]):
            blbl = base_labels[i]
            if blbl not in seen_base:
                seen_base[blbl] = i
        rep_idx = list(seen_base.values())

        rep_x   = [xs[i]           for i in rep_idx]
        rep_y   = [ys[i]           for i in rep_idx]
        rep_lbl = [ts_names[i]     for i in rep_idx]  # ts-XXX label

        # Colour by mean n_tilts across all sub-TS of that base Position
        base_to_nts = defaultdict(list)
        for i in idx:
            base_to_nts[base_labels[i]].append(n_tilts_list[i])
        rep_nt = [float(np.mean(base_to_nts[base_labels[i]])) for i in rep_idx]

        # ── Collect per-TS stats for histograms ───────────────────────
        rots, alphas, n_tilts_c, n_flagged_c, resolutions = [], [], [], [], []
        thicknesses = []
        cluster_angpix = None
        for ts in ts_in_cluster:
            d = all_ts[ts]
            frames = d['frames']
            if not frames:
                continue
            rots.append(frames[0].get('rot'))
            alphas.append(d.get('alpha_offset'))
            n_tilts_c.append(len(frames))
            n_flagged_c.append(sum(1 for f in frames if f.get('is_flagged', False)))
            res_vals = [f['fit_spacing_A'] for f in frames
                        if f.get('fit_spacing_A') is not None]
            resolutions.append(float(np.median(res_vals)) if res_vals else None)
            if d.get('thickness') is not None:
                thicknesses.append(d['thickness'])
            if cluster_angpix is None and d.get('angpix') is not None:
                cluster_angpix = d['angpix']

        # ── Figure: scatter (left, full height) + 2×3 histograms (right) ─
        fig = plt.figure(figsize=(18, 9), constrained_layout=True)
        gs  = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1])
        ax_s  = fig.add_subplot(gs[:, 0])
        ax_h  = [fig.add_subplot(gs[r, col]) for r in range(2) for col in range(1, 4)]

        fig.suptitle(f'Lamella {c+1}  —  {len(ts_in_cluster)} tilt series',
                     fontsize=13, fontweight='bold')

        # Scatter
        sc = ax_s.scatter(rep_x, rep_y, c=rep_nt, cmap='viridis',
                          vmin=vmin, vmax=vmax,
                          s=100, edgecolors='white', linewidths=0.6, zorder=3)
        plt.colorbar(sc, ax=ax_s, label='mean # aligned tilts')
        for px, py, plbl in zip(rep_x, rep_y, rep_lbl):
            ax_s.annotate(plbl, xy=(px, py), xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=8, fontweight='bold', zorder=5)
        ax_s.set_title('Lamella position on stage', fontsize=10)
        ax_s.set_xlabel('Stage X (µm)')
        ax_s.set_ylabel('Stage Y (µm)')
        ax_s.set_aspect('equal', adjustable='datalim')
        ax_s.grid(True, lw=0.4, alpha=0.4)

        # Histograms
        _hist_panel(ax_h[0], rots,        'ROT (°)',                    'mediumpurple', '.2f', '°')
        _hist_panel(ax_h[1], alphas,       'AlphaOffset (°)',            'teal',         '.2f', '°')
        _hist_panel(ax_h[2], thicknesses,  'AlignZ (px)', 'slateblue', '.0f', ' px')
        if cluster_angpix is not None and thicknesses:
            cur = ax_h[2].get_title()
            ax_h[2].set_title(cur + f'\n(angpix = {cluster_angpix} Å/px)',
                              fontsize=7, color='steelblue', fontweight='bold')
        _hist_panel(ax_h[3], n_tilts_c,   '# aligned tilts',
                    'darkorange', '.0f', '', show_suggested=False)
        _hist_panel(ax_h[4], n_flagged_c, '# flagged frames',
                    'tomato',     '.0f', '', show_suggested=False)
        _hist_panel(ax_h[5], resolutions, 'Resolution — fit spacing (Å)',
                    'goldenrod',  '.1f', ' Å', show_suggested=False)

        lam_png = f'stage_positions_lamella_{c+1:02d}.png'
        fig.savefig(out_dir / lam_png, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'  Lamella {c+1:2d}       : {out_dir / lam_png}')
        output_entries.append((f'[Lamella {c+1}] Stage Positions', lam_png))

        # Accumulate per-lamella suggested values
        def _safe_median(vals):
            clean = [v for v in vals if v is not None]
            return round(float(np.median(clean)), 2) if clean else None

        align_z_px  = int(round(np.median([t for t in thicknesses if t is not None]))) \
                      if any(t is not None for t in thicknesses) else None
        align_z_nm  = round(align_z_px * cluster_angpix / 10, 1) \
                      if (align_z_px is not None and cluster_angpix is not None) else None
        lamella_stats[str(c + 1)] = {
            'rot_deg':          _safe_median(rots),
            'alpha_offset_deg': _safe_median(alphas),
            'align_z_px':       align_z_px,
            'align_z_nm':       align_z_nm,
            'angpix':           cluster_angpix,
            'n_ts':             len(ts_in_cluster),
        }

    return output_entries, lamella_stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI integration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'analyse',
        help='Parse AreTomo .aln files and produce diagnostic plots and HTML viewer',
        formatter_class=__import__('argparse').ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',     '-i', default='run001',
                   help='Directory containing .aln files')
    p.add_argument('--output',    '-o', default='run001_analysis',
                   help='Output directory for plots, JSON, TSV, and HTML')
    p.add_argument('--threshold', '-t', type=float, default=80.0,
                   help='%% overlap below which a frame is flagged')
    p.add_argument('--angpix',    '-a', type=float, default=None,
                   help='Pixel size in Å/px — used to convert thickness from '
                        'pixels to nm.  Auto-read from project.json (mdoc_data) '
                        'if omitted.')
    p.add_argument('--n-lamellae', '-n', type=int, default=None,
                   help='Expected number of lamellae on the grid.  When set, '
                        'K-means clustering is applied to the stage X-Y '
                        'positions and lamella groups are colour-coded in the '
                        'stage position scatter plot (requires scikit-learn).')
    p.add_argument('--reuse-lamellae', action='store_true',
                   help='Deprecated — lamella assignments are now reused automatically '
                        'whenever lamella_positions.csv exists in --output.  '
                        'Flag is kept for backward compatibility.')
    p.add_argument('--refit-lamellae', action='store_true',
                   help='Ignore saved lamella assignments in project.json and '
                        're-run K-means clustering from scratch.  Requires '
                        '--n-lamellae.  Overwrites the saved assignments.')
    p.add_argument('--compare-previous', '-C', default=None,
                   help='Output directory from a previous analyse run.  '
                        'Previous run data is shown in grey in per-TS overlap '
                        'plots and global summary histograms.')
    p.set_defaults(func=run)
    return p


def run(args):
    # Force line-buffered stdout so output appears immediately when piped (e.g. | tee)
    sys.stdout.reconfigure(line_buffering=True)

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    aln_files = sorted(in_dir.glob('*.aln'))
    if not aln_files:
        print(f'No .aln files found in {in_dir}')
        return

    if args.angpix is None:
        args.angpix = get_angpix()

    # ── TLT directory: check --input first (direct cmd=0 case), then project.json ──
    if any(in_dir.glob('*_TLT.txt')):
        tlt_dir = in_dir
    else:
        _proj_tlt = get_tlt_dir()
        if _proj_tlt is not None and _proj_tlt.exists() and any(_proj_tlt.glob('*_TLT.txt')):
            tlt_dir = _proj_tlt
            print(f'Note: using cmd=0 TLT dir from project.json → {tlt_dir}\n')
        else:
            _hint = (f'\n       (project.json tlt_dir {_proj_tlt} also has no _TLT.txt files)'
                     if _proj_tlt is not None else '')
            print(f'ERROR: no _TLT.txt files found in {in_dir}.{_hint}')
            print(f'       TLT files are required for dose, z_value, and stage position enrichment.')
            print(f'       Register the cmd=0 output directory:')
            print(f'         aretomo3-preprocess enrich --tlt-data <cmd0-output-dir>')
            sys.exit(1)

    # ── Load project.json — mdoc cache, lamella assignments, MRC stack paths ──
    _proj        = _load_project()
    _cached_mdoc = _proj.get('mdoc_data', {}).get('per_ts')
    if not _cached_mdoc:
        print(f'ERROR: mdoc_data not found in project.json.')
        print(f'       Run validate-mdoc first to populate it, or register manually:')
        print(f'         aretomo3-preprocess enrich --mdoc-data <frames/>')
        sys.exit(1)

    _stacks        = _proj.get('input_stacks', {}).get('stacks', {})
    _n_mdoc_cached = 0

    # Build ts-name → original mdoc stem mapping from rename_ts lookup.
    # mdoc_data is keyed by original filename stems (e.g. 'Position_1') while
    # .aln files are named ts-xxx, so a direct lookup would always miss.
    _rename_lookup = _proj.get('rename_ts', {}).get('lookup', {})
    _ts_to_mdoc_stem = {
        Path(ts_mdoc).stem: Path(orig_path).stem
        for ts_mdoc, orig_path in _rename_lookup.items()
    }  # e.g. {'ts-001': 'Position_1', 'ts-002': 'Position_10', ...}

    angpix_str = f'{args.angpix} Å/px' if args.angpix else 'from mdoc_data'
    print(f'Found {len(aln_files)} .aln files  |  threshold = {args.threshold}%  '
          f'|  pixel size = {angpix_str}  |  TLT dir = {tlt_dir}\n')

    # ── Pass 1: parse all files, attach overlap + CTF + TLT + mdoc ───────────
    all_ts = {}
    _t1_start = time.perf_counter()
    _timing   = {'aln': 0.0, 'ctf': 0.0, 'tlt': 0.0, 'mdoc': 0.0,
                 'enrich': 0.0, 'mrc_val': 0.0}

    for aln_path in _tqdm(aln_files, desc='Pass 1 — parsing', unit='TS', ncols=80):
        ts_name = aln_path.stem

        _t = time.perf_counter()
        data    = parse_aln_file(aln_path)
        _timing['aln'] += time.perf_counter() - _t

        if not data['frames'] or data['width'] is None:
            print(f'  WARNING: {ts_name} could not be parsed — skipping')
            continue

        W, H = data['width'], data['height']

        _t = time.perf_counter()
        ctf_path = aln_path.parent / f'{ts_name}_CTF.txt'
        ctf_data = parse_ctf_file(ctf_path) if ctf_path.exists() else {}
        _timing['ctf'] += time.perf_counter() - _t

        _t = time.perf_counter()
        tlt_path = tlt_dir / f'{ts_name}_TLT.txt'
        tlt_data = parse_tlt_file(tlt_path) if tlt_path.exists() else {}
        _timing['tlt'] += time.perf_counter() - _t

        _t = time.perf_counter()
        _mdoc_key   = _ts_to_mdoc_stem.get(ts_name, ts_name)
        _cm         = _cached_mdoc.get(_mdoc_key, {})
        mdoc_data   = {int(k): v for k, v in _cm.get('frames', {}).items()}
        mdoc_angpix = _cm.get('angpix')
        if _cm:
            _n_mdoc_cached += 1
        _timing['mdoc'] += time.perf_counter() - _t

        # Resolve pixel size: CLI arg → mdoc PixelSpacing → None
        angpix = args.angpix if args.angpix is not None else mdoc_angpix
        data['angpix'] = angpix
        if data['thickness'] is not None and angpix is not None:
            data['thickness_nm'] = round(data['thickness'] * angpix / 10.0, 2)
        else:
            data['thickness_nm'] = None

        _t = time.perf_counter()

        # Cumulative dose (RELION prior-dose convention: first acquired = 0)
        cum_dose_by_acq = {}
        if tlt_data:
            sorted_by_acq = sorted(tlt_data.values(), key=lambda r: r['acq_order'])
            running = 0.0
            for r in sorted_by_acq:
                cum_dose_by_acq[r['acq_order']] = running
                running += r['dose_e_per_A2']

        _MDOC_KEYS = ('tilt_angle', 'sub_frame_path', 'mdoc_defocus',
                      'target_defocus', 'datetime', 'stage_x', 'stage_y',
                      'stage_z', 'exposure_time', 'num_subframes')

        for f in data['frames']:
            f['overlap_pct']  = compute_overlap(f['tx'], f['ty'], W, H,
                                                   rot_deg=f['rot'])
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

            tlt = tlt_data.get(f['sec'], {})
            f['nominal_tilt']             = tlt.get('nominal_tilt')
            f['acq_order']                = tlt.get('acq_order')
            f['dose_e_per_A2']            = tlt.get('dose_e_per_A2')
            f['z_value']                  = tlt.get('z_value')
            f['cumulative_dose_e_per_A2'] = (
                round(cum_dose_by_acq[tlt['acq_order']], 2)
                if tlt.get('acq_order') is not None else None
            )

            mdoc = mdoc_data.get(f['z_value'], {}) if f['z_value'] is not None else {}
            for k in _MDOC_KEYS:
                f[k] = mdoc.get(k)

        # Dark frame enrichment
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

        _timing['enrich'] += time.perf_counter() - _t

        # Sanity checks (MRC path from input_stacks if registered; silently skipped if absent)
        _t = time.perf_counter()
        _mrc_info = _stacks.get(ts_name)
        mrc_path  = Path(_mrc_info['path']) if _mrc_info else None
        data['warnings'] = _validate_ts(data, tlt_data, mdoc_data, mrc_path)
        _timing['mrc_val'] += time.perf_counter() - _t

        all_ts[ts_name] = data

    # ── Pass 1 timing summary ─────────────────────────────────────────────────
    _t1_total = time.perf_counter() - _t1_start
    _t1_other = _t1_total - sum(_timing.values())
    print(f'\nPass 1 complete  ({len(all_ts)} TS in {_t1_total:.1f}s)\n')
    _steps = [('aln parse',  _timing['aln']),
              ('ctf parse',  _timing['ctf']),
              ('tlt parse',  _timing['tlt']),
              ('mdoc parse', _timing['mdoc']),
              ('enrichment', _timing['enrich']),
              ('mrc header', _timing['mrc_val']),
              ('other',      _t1_other)]
    for _label, _secs in _steps:
        _pct  = 100 * _secs / _t1_total if _t1_total > 0 else 0
        _bar  = '█' * int(_pct / 2.5)
        print(f'  {_label:<12}  {_secs:6.2f}s  {_pct:5.1f}%  {_bar}')
    print(f'  mdoc: {_n_mdoc_cached} / {len(all_ts)} TS enriched from project.json')
    print()

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

    # ── Load previous run data for comparison ────────────────────────────────
    prev_all_ts = {}
    if args.compare_previous:
        prev_json = Path(args.compare_previous) / 'alignment_data.json'
        if prev_json.exists():
            with open(prev_json) as fh:
                prev_all_ts = json.load(fh)
            print(f'Comparing with previous run: {prev_json}  ({len(prev_all_ts)} TS)\n')
        else:
            print(f'WARNING: --compare-previous: {prev_json} not found'
                  f' — comparison skipped\n')

    # ── Pass 2: generate console output, JSON, TSV, plots ────────────────────
    sep           = '─' * 70
    all_parsed    = {}
    flagged_rows  = []
    ts_entries    = []
    total_flagged = 0
    n_ts          = len(all_ts)
    _t2_start     = time.perf_counter()
    _t2_plot      = 0.0

    _p2     = _tqdm(list(all_ts.items()), desc='Pass 2 — plots', unit='TS', ncols=80)
    _pwrite = getattr(_p2, 'write', print)  # tqdm.write keeps bar intact; fallback to print

    for i_ts, (ts_name, data) in enumerate(_p2, 1):
        bad_frames    = [f for f in data['frames'] if f['is_flagged']]
        n_bad         = len(bad_frames)
        total_flagged += n_bad

        status = '✗' if n_bad else '✓'
        _pwrite(sep)
        _pwrite(f'  {status}  {ts_name}   {n_bad} frame(s) below {args.threshold}%  '
                f'| {len(data["dark_frames"])} dark frame(s)')
        for w in data.get('warnings', []):
            _pwrite(f'  WARN  {w}')
        if bad_frames:
            _pwrite(f'     {"SEC":>4}  {"Tilt (°)":>9}  {"TX (px)":>10}  '
                    f'{"TY (px)":>10}  {"Overlap":>8}')
            for f in bad_frames:
                _pwrite(f'     {f["sec"]:>4}  {f["tilt"]:>9.2f}  {f["tx"]:>10.1f}  '
                        f'{f["ty"]:>10.1f}  {f["overlap_pct"]:>7.1f}%')
                flagged_rows.append({
                    'ts':          ts_name,
                    'sec':         f['sec'],
                    'tilt':        f['tilt'],
                    'tx':          f['tx'],
                    'ty':          f['ty'],
                    'overlap_pct': f['overlap_pct'],
                })

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

        png_name = f'{ts_name}.png'
        vol_path = in_dir / f'{ts_name}_Vol.mrc'
        _tp = time.perf_counter()
        plot_tilt_series(ts_name, data, args.threshold,
                         str(out_dir / png_name), global_ranges,
                         prev_data=prev_all_ts.get(ts_name),
                         vol_path=vol_path if vol_path.exists() else None)
        _t2_plot += time.perf_counter() - _tp

        ts_entries.append({
            'name':     ts_name,
            'png':      png_name,
            'n_bad':    n_bad,
            'n_frames': len(data['frames']),
            'n_dark':   len(data['dark_frames']),
        })

    print(sep)

    # ── Global summary plot ───────────────────────────────────────────────────
    _tp = time.perf_counter()
    summary_png = 'global_summary.png'
    plot_global_summary(all_ts, args.threshold, global_ranges,
                        str(out_dir / summary_png),
                        prev_ts=prev_all_ts or None)
    _t2_summary = time.perf_counter() - _tp
    ts_entries.insert(0, {
        'name':     '[Summary] Global Analysis',
        'png':      summary_png,
        'n_bad':    0,
        'n_frames': sum(len(d['frames']) for d in all_ts.values()),
        'n_dark':   0,
    })

    # ── Stage positions scatter plot ──────────────────────────────────────────
    # ts-name → original-path lookup for stage position labels (from rename-ts)
    stage_lookup = _proj.get('rename_ts', {}).get('lookup', {})

    # Lamella assignments locked in project.json — reuse automatically.
    # First run: no saved assignments → K-means (requires --n-lamellae).
    cluster_ids_override = None
    csv_path             = out_dir / 'lamella_positions.csv'
    _saved_positions     = _proj.get('lamella_assignments', {}).get('positions', {})

    refit = getattr(args, 'refit_lamellae', False)
    if refit and args.n_lamellae is None:
        print('ERROR: --refit-lamellae requires --n-lamellae N')
        sys.exit(1)

    if _saved_positions and not refit:
        cluster_ids_override = {ts: pos - 1 for ts, pos in _saved_positions.items()}
        print(f'  Lamella assignments locked — from project.json'
              f'  ({len(cluster_ids_override)} TS)')
        if args.n_lamellae is not None:
            print(f'  WARNING: --n-lamellae {args.n_lamellae} is ignored because '
                  f'lamella assignments are already locked in project.json.\n'
                  f'           Use --refit-lamellae to re-cluster.')
    elif _saved_positions and refit:
        print(f'  --refit-lamellae: ignoring saved assignments, re-running K-means '
              f'with --n-lamellae {args.n_lamellae}')

    if cluster_ids_override is not None:
        new_ts = [ts for ts in all_ts if ts not in cluster_ids_override]
        if new_ts:
            print(f'  WARNING: {len(new_ts)} TS not in saved lamella assignments'
                  f' (assigned to lamella 1 as fallback):')
            for ts in sorted(new_ts):
                print(f'    {ts}')
    else:
        # No saved assignments exist yet — require the user to declare the
        # number of lamellae so assignments are established intentionally.
        if args.n_lamellae is None:
            print(
                f'\nERROR: No lamella assignments found for this project.\n'
                f'       Please re-run with --n-lamellae N to cluster the\n'
                f'       {len(all_ts)} tilt series into N lamellae, e.g.:\n\n'
                f'         aretomo3-preprocess analyse ... --n-lamellae 6\n\n'
                f'       Assignments will be saved to project.json and locked\n'
                f'       for all future runs.\n'
            )
            sys.exit(1)

    _tp = time.perf_counter()
    stage_entries, lamella_stats = plot_stage_positions(
        all_ts,
        out_dir              = out_dir,
        n_lamellae           = args.n_lamellae,
        lookup               = stage_lookup or None,
        cluster_ids_override = cluster_ids_override,
    )
    _t2_stage = time.perf_counter() - _tp

    # Save lamella assignments to project.json.
    # Normally written once (update_section_once); --refit-lamellae forces overwrite.
    if csv_path.exists() and ('lamella_assignments' not in _proj or refit):
        _la_positions = {}
        with open(csv_path, newline='') as fh:
            for row in csv.DictReader(fh):
                _la_positions[row['ts_name']] = int(row['lamella'])
        _la_data = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'n_ts':      len(_la_positions),
            'positions': _la_positions,   # ts_name -> lamella number (1-indexed)
        }
        if refit:
            update_section('lamella_assignments', _la_data)
        else:
            update_section_once('lamella_assignments', _la_data)

    total_frames = sum(len(d['frames']) for d in all_ts.values())
    for i, (entry_name, entry_png) in enumerate(stage_entries):
        ts_entries.insert(1 + i, {
            'name':     entry_name,
            'png':      entry_png,
            'n_bad':    0,
            'n_frames': total_frames,
            'n_dark':   0,
        })

    # JSON
    _tp = time.perf_counter()
    json_path = out_dir / 'alignment_data.json'
    with open(json_path, 'w') as fh:
        json.dump(all_parsed, fh, indent=2)
    _t2_json = time.perf_counter() - _tp

    # Flagged TSV
    tsv_path = out_dir / 'flagged_frames.tsv'
    with open(tsv_path, 'w') as fh:
        fh.write('ts_name\tsec\ttilt\ttx\tty\toverlap_pct\n')
        for row in flagged_rows:
            fh.write(
                f'{row["ts"]}\t{row["sec"]}\t{row["tilt"]:.2f}\t'
                f'{row["tx"]:.3f}\t{row["ty"]:.3f}\t{row["overlap_pct"]:.2f}\n'
            )

    # ── Load gain-check results from project.json (automatic) ────────────────
    gain_check = _proj.get('gain_check')
    if gain_check is not None:
        gc_dir = get_gain_check_dir()
        if gc_dir is not None:
            # Copy gain-check PNGs into the output directory so the HTML is
            # self-contained (relative src= paths work from any location).
            for png_name in ('corrected_averages.png', 'cv_vs_nmovies.png'):
                src = gc_dir / png_name
                if src.exists():
                    shutil.copy2(src, out_dir / png_name)
                else:
                    print(f'Note: gain-check PNG not found: {src}')
            print(f'Gain check results loaded from project.json (dir: {gc_dir})\n')

    # ── Load TS selection from project.json (automatic) ──────────────────────
    sel_section = _proj.get('select_ts', {})
    if sel_section.get('ts_names') is not None:
        selection = {ts: True  for ts in sel_section['ts_names']}
        # Mark all TS in alignment_data that are NOT in the selection as False
        for ts in all_ts:
            if ts not in selection:
                selection[ts] = False
        n_sel = sum(1 for v in selection.values() if v)
        print(f'TS selection loaded from project.json: '
              f'{n_sel}/{len(all_ts)} selected\n')
    else:
        selection = None

    # ── Load ratings from ts_ratings.csv if present ───────────────────────────
    ratings_csv = out_dir / 'ts_ratings.csv'
    saved_ratings = {}
    if ratings_csv.exists():
        import csv as _csv
        with open(ratings_csv, newline='') as _fh:
            for row in _csv.DictReader(_fh):
                try:
                    saved_ratings[row['ts_name']] = int(row['rating'])
                except (KeyError, ValueError):
                    pass
        print(f'Loaded {len(saved_ratings)} ratings from {ratings_csv.name}')

    # HTML
    _tp = time.perf_counter()
    html_path = out_dir / 'index.html'
    make_html(ts_entries, str(html_path), args.threshold, gain_check, selection,
              ratings=saved_ratings or None)
    _t2_html = time.perf_counter() - _tp

    # ── Project JSON ──────────────────────────────────────────────────────────
    n_ts_bad  = sum(1 for e in ts_entries if e['n_bad'] > 0)
    n_ts_warn = sum(1 for d in all_ts.values() if d.get('warnings'))

    # ── Compute global suggested values ───────────────────────────────────────
    all_rots        = [d['frames'][0]['rot']   for d in all_ts.values() if d['frames']]
    all_alphas      = [d['alpha_offset']        for d in all_ts.values()
                       if d.get('alpha_offset') is not None]
    all_thicknesses = [d['thickness']           for d in all_ts.values()
                       if d.get('thickness') is not None]
    all_angpix_vals = [d['angpix']             for d in all_ts.values()
                       if d.get('angpix') is not None]

    from collections import Counter as _Counter
    angpix_global = (float(_Counter(all_angpix_vals).most_common(1)[0][0])
                     if all_angpix_vals else None)

    def _gmed(vals):
        return round(float(np.median(vals)), 2) if vals else None

    global_align_z_px = int(round(np.median(all_thicknesses))) if all_thicknesses else None
    global_suggested = {
        'rot_deg':          _gmed(all_rots),
        'alpha_offset_deg': _gmed(all_alphas),
        'align_z_px':       global_align_z_px,
        'align_z_nm':       (round(global_align_z_px * angpix_global / 10, 1)
                             if (global_align_z_px and angpix_global) else None),
        'angpix':           angpix_global,
    }
    recommended_tilt_axis = global_suggested['rot_deg']

    print()
    update_section(
        section    = 'analyse',
        values     = {
            'command':                ' '.join(sys.argv),
            'args':                   args_to_dict(args),
            'timestamp':              datetime.datetime.now().isoformat(timespec='seconds'),
            'n_tilt_series':          len(all_ts),
            'n_flagged_frames':       total_flagged,
            'n_ts_with_flags':        n_ts_bad,
            'n_ts_with_warnings':     n_ts_warn,
            'output_dir':             str(out_dir),
            'recommended_tilt_axis':  recommended_tilt_axis,
            'global_suggested':       global_suggested,
            'lamella_suggested':      lamella_stats,
        },
        backup_dir = out_dir,
    )

    # Summary
    print(f'\nSummary')
    print(f'  Tilt series processed : {len(all_ts)}')
    print(f'  TS with flagged frames: {n_ts_bad}')
    print(f'  Total flagged frames  : {total_flagged}')
    print(f'  TS with sanity warnings: {n_ts_warn}')
    if recommended_tilt_axis is not None:
        print(f'  Recommended tilt axis : {recommended_tilt_axis}°  '
              f'(median ROT, use as --tilt-axis for run002)')
    if global_suggested.get('align_z_px') is not None:
        apx_note = (f'  [{global_suggested["align_z_nm"]} nm at {global_suggested["angpix"]} Å/px]'
                    if global_suggested.get('align_z_nm') else '')
        print(f'  Recommended AlignZ    : {global_suggested["align_z_px"]} px'
              f'{apx_note}  (use as --align-z for run002)')
    if global_suggested.get('alpha_offset_deg') is not None:
        print(f'  Median AlphaOffset    : {global_suggested["alpha_offset_deg"]}°  '
              f'(informational; AreTomo3 estimates this automatically)')
    print(f'\nOutput')
    print(f'  Plots          : {out_dir}/<ts-name>.png')
    print(f'  Stage positions: {out_dir}/stage_positions*.png')
    print(f'  HTML viewer    : {html_path}')
    print(f'  Alignment JSON : {json_path}')
    print(f'  Flagged TSV    : {tsv_path}')

    # ── Timing summary ────────────────────────────────────────────────────────
    _t2_total  = time.perf_counter() - _t2_start
    _t2_other  = _t2_total - _t2_plot - _t2_summary - _t2_stage - _t2_json - _t2_html
    _t_overall = _t1_total + _t2_total
    print(f'\nTiming summary  (total {_t_overall:.1f}s)')
    print(f'  {"Pass 1 — parsing":<28}  {_t1_total:6.1f}s')
    for _label, _secs in _steps:
        print(f'    {_label:<26}  {_secs:6.2f}s  ({100*_secs/_t1_total:.0f}%)')
    print(f'  {"Pass 2 — output":<28}  {_t2_total:6.1f}s')
    for _label, _secs in [('per-TS plots',    _t2_plot),
                           ('global summary',  _t2_summary),
                           ('stage positions', _t2_stage),
                           ('HTML',            _t2_html),
                           ('JSON write',      _t2_json),
                           ('other',           _t2_other)]:
        print(f'    {_label:<26}  {_secs:6.2f}s  ({100*_secs/_t2_total:.0f}%)')
