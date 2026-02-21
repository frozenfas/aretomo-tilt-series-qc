"""
analyse subcommand — parse AreTomo .aln files, produce per-TS plots,
a global summary PNG, an HTML viewer, alignment JSON, and flagged TSV.
"""

import sys
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
    import mdocfile as _mdocfile
    _HAS_MDOCFILE = True
except ImportError:
    _HAS_MDOCFILE = False

from aretomo3_editor.shared.project_json import update_section, args_to_dict
from aretomo3_editor.shared.parsers import (
    parse_aln_file, parse_ctf_file, parse_tlt_file, parse_mdoc_file,
    _float_or_none,
)
from aretomo3_editor.shared.geometry import compute_overlap, rotated_rect_corners
from aretomo3_editor.shared.colours import (
    OVL_CMAP, OVL_NORM, RES_CMAP,
    _ovl_colour, _ovl_sm, _res_sm,
)


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
    tilt_xlim    = (global_ranges['tilt_min'],    global_ranges['tilt_max'])
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
            with mrcfile.open(mrc_path, permissive=True) as m:
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
    n_ts = len(all_ts)

    # Per-TS accumulators
    n_total_list      = []
    n_aligned_list    = []
    n_passing_list    = []
    n_flagged_list    = []
    rot_list          = []
    alpha_offset_list = []
    thickness_list    = []
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

    # ── (2,1) Lamella Thickness AlignZ ────────────────────────────────────
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

def make_html(ts_entries, out_path, threshold, gain_check=None):
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
    """
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
            marker      = '  \u2190 best' if name == best else ''
            style       = 'color:#66bb6a;font-weight:bold' if name == best else ''
            ssim_flat_s = f"{s['ssim_vs_flat']:.4f}" if s.get('ssim_vs_flat') is not None else 'n/a'
            ssim_raw_s  = f"{s['ssim_vs_raw']:.4f}"  if s.get('ssim_vs_raw')  is not None else 'n/a'
            rows_html += (
                f'<tr style="{style}"><td>{name}{marker}</td>'
                f'<td>{s["cv"]:.4f}</td><td>{ssim_flat_s}</td><td>{ssim_raw_s}</td></tr>\n'
            )

        gain_tab_btn  = '<button class="tab-btn active" onclick="switchTab(\'gain\')">Gain Transform Check</button>'
        ts_tab_btn    = '<button class="tab-btn" onclick="switchTab(\'ts\')">Tilt Series Analysis</button>'
        gain_section  = f"""
  <div id="tab-gain" class="tab-section">
    <div class="gc-card">
      <div class="gc-best">Best transform: {best}</div>
      <div class="gc-flags">{flags}</div>
      <table class="gc-table">
        <tr><th>Transform</th><th>CV &#x2193; (lower = flatter)</th><th>SSIM vs flat &#x2191;</th><th>SSIM vs raw &#x2191;</th></tr>
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
    </div>

    <div id="progress"><div id="progress-bar"></div></div>
    <div id="title"></div>

    <div id="img-wrap">
      <img id="main-img" src="" alt="tilt series plot">
    </div>

    <p id="hint">Keyboard: &#8592; &#8594; to navigate between tilt series</p>
  </div>

  <script>
    {tab_init_js}
    {tab_switch_js}

    const images  = {images_js};
    const titles  = {titles_js};
    const n       = images.length;
    let   idx     = 0;

    const img  = document.getElementById('main-img');
    const sel  = document.getElementById('ts-select');
    const ctr  = document.getElementById('counter');
    const ttl  = document.getElementById('title');
    const pbar = document.getElementById('progress-bar');

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
    p.add_argument('--mdocdir',   '-m', default='frames',
                   help='Directory containing per-TS .mdoc files '
                        '(ts-xxx.mdoc expected; skip enrichment if absent)')
    p.add_argument('--angpix',    '-a', type=float, default=None,
                   help='Pixel size in Å/px — used to convert thickness from '
                        'pixels to nm.  If omitted, read from mdoc PixelSpacing.')
    p.add_argument('--mrcdir',    '-r', default=None,
                   help='Directory containing ts-xxx.mrc raw stacks for MRC '
                        'header sanity checks (nx/ny/nz vs .aln).  '
                        'Defaults to --input; per-TS check silently skipped '
                        'if the .mrc file is not found.')
    p.add_argument('--gain-check-dir', '-g', default=None,
                   help='Output directory from a previous check-gain-transform '
                        'run (must contain aretomo3_project.json backup, '
                        'corrected_averages.png, cv_vs_nmovies.png).  When '
                        'provided, a "Gain Transform Check" tab is prepended '
                        'to the HTML report.')
    p.set_defaults(func=run)
    return p


def run(args):
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

    # MRC directory: explicit --mrcdir, else same directory as the .aln files.
    # Per-TS MRC checks are silently skipped if the .mrc file is absent.
    mrc_dir = Path(args.mrcdir) if args.mrcdir else in_dir
    if args.mrcdir and not mrc_dir.exists():
        print(f'WARNING: --mrcdir {mrc_dir} not found — MRC checks skipped\n')
        mrc_dir = None

    angpix_str = f'{args.angpix} Å/px' if args.angpix else 'from mdoc (or None)'
    print(f'Found {len(aln_files)} .aln files  |  threshold = {args.threshold}%  '
          f'|  pixel size = {angpix_str}\n')

    # ── Pass 1: parse all files, attach overlap + CTF + TLT + mdoc ───────────
    all_ts = {}

    for aln_path in aln_files:
        ts_name = aln_path.stem
        data    = parse_aln_file(aln_path)

        if not data['frames'] or data['width'] is None:
            print(f'  WARNING: {ts_name} could not be parsed — skipping')
            continue

        W, H = data['width'], data['height']

        ctf_path = aln_path.parent / f'{ts_name}_CTF.txt'
        ctf_data = parse_ctf_file(ctf_path) if ctf_path.exists() else {}

        tlt_path = aln_path.parent / f'{ts_name}_TLT.txt'
        tlt_data = parse_tlt_file(tlt_path) if tlt_path.exists() else {}

        mdoc_path = mdoc_dir / f'{ts_name}.mdoc'
        mdoc_data = (parse_mdoc_file(mdoc_path)
                     if _HAS_MDOCFILE and mdoc_path.exists() else {})

        # Resolve pixel size: CLI arg → mdoc PixelSpacing → None
        angpix = args.angpix
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

        # Sanity checks
        mrc_path = (mrc_dir / f'{ts_name}.mrc') if mrc_dir else None
        data['warnings'] = _validate_ts(data, tlt_data, mdoc_data, mrc_path)

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

        status = '✗' if n_bad else '✓'
        print(sep)
        print(f'  {status}  {ts_name}   {n_bad} frame(s) below {args.threshold}%  '
              f'| {len(data["dark_frames"])} dark frame(s)')
        for w in data.get('warnings', []):
            print(f'  WARN  {w}')
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

    # ── Load gain-check results (optional) ───────────────────────────────────
    gain_check = None
    if args.gain_check_dir:
        gc_dir = Path(args.gain_check_dir)
        gc_json = gc_dir / 'aretomo3_project.json'
        if not gc_json.exists():
            print(f'WARNING: --gain-check-dir {gc_dir} has no aretomo3_project.json — '
                  f'gain check tab skipped\n')
        else:
            with open(gc_json) as fh:
                proj = json.load(fh)
            gain_check = proj.get('gain_check')
            # Copy gain-check PNGs into the output directory so the HTML is
            # self-contained (relative src= paths work from any location).
            for png_name in ('corrected_averages.png', 'cv_vs_nmovies.png'):
                src = gc_dir / png_name
                if src.exists():
                    shutil.copy2(src, out_dir / png_name)
                else:
                    print(f'WARNING: gain-check PNG not found: {src}')
            print(f'Gain check results loaded from {gc_dir}\n')

    # HTML
    html_path = out_dir / 'index.html'
    make_html(ts_entries, str(html_path), args.threshold, gain_check)

    # ── Project JSON ──────────────────────────────────────────────────────────
    n_ts_bad  = sum(1 for e in ts_entries if e['n_bad'] > 0)
    n_ts_warn = sum(1 for d in all_ts.values() if d.get('warnings'))
    print()
    update_section(
        section    = 'analyse',
        values     = {
            'command':              ' '.join(sys.argv),
            'args':                 args_to_dict(args),
            'timestamp':            datetime.datetime.now().isoformat(timespec='seconds'),
            'n_tilt_series':        len(all_ts),
            'n_flagged_frames':     total_flagged,
            'n_ts_with_flags':      n_ts_bad,
            'n_ts_with_warnings':   n_ts_warn,
            'output_dir':           str(out_dir),
        },
        backup_dir = out_dir,
    )

    # Summary
    print(f'\nSummary')
    print(f'  Tilt series processed : {len(ts_entries)}')
    print(f'  TS with flagged frames: {n_ts_bad}')
    print(f'  Total flagged frames  : {total_flagged}')
    print(f'  TS with sanity warnings: {n_ts_warn}')
    print(f'\nOutput')
    print(f'  Plots          : {out_dir}/<ts-name>.png')
    print(f'  HTML viewer    : {html_path}')
    print(f'  Alignment JSON : {json_path}')
    print(f'  Flagged TSV    : {tsv_path}')
