"""
check-gain-transform — determine the correct AreTomo3 gain correction
parameters (-RotGain / -FlipGain) for K3 TIFF movies + MRC gain files.

Scope
-----
This command is specific to:
  • Rectangular (K3) detectors  — square sensors are not yet supported
  • TIFF movie format            — EER support is planned for a future version
  • MRC gain reference           — must be 32-bit float (mode 2)
  • AreTomo3 conventions         — flip/rotation semantics match AreTomo3 source

Algorithm
---------
Four dimension-preserving transforms are tested:
  none   (-RotGain 0 -FlipGain 0)
  flipud (-RotGain 0 -FlipGain 1)   flip around horizontal axis
  fliplr (-RotGain 0 -FlipGain 2)   flip around vertical axis
  rot180 (-RotGain 2 -FlipGain 0)   180° rotation

MRC Y-convention (AreTomo3)
---------------------------
AreTomo3 reads MRC gain files with an implicit Y-flip (MRC standard: row 0 =
bottom of image, opposite to numpy convention).  With --software aretomo3
(default), the loaded gain is pre-flipped with flipud() before scoring.  This
makes the transform table above a direct 1:1 mapping to AreTomo3 flags:

  none   → raw × flipud(G_mrc)           → -FlipGain 0  (MRC implicit flip)
  flipud → raw × flipud(flipud(G_mrc))   → -FlipGain 1  (cancels MRC flip)
  fliplr → raw × fliplr(flipud(G_mrc))   → -FlipGain 2
  rot180 → raw × rot180(flipud(G_mrc))   → -RotGain 2 -FlipGain 0

Without pre-flipping (legacy behaviour, --software none) the reported flags
would be inverted — "flipud" best would incorrectly suggest -FlipGain 1.

For each selected movie the sub-frames are summed (float64) and multiplied
by each transformed gain.  After accumulating all movies the best transform
is the one that produces the flattest corrected image, quantified by the
coefficient of variation (CV = std / mean).  Lower CV → flatter → better.

Movie selection
---------------
Movies are selected by acquisition order (parsed from the filename), NOT by
tilt angle.  For pre-tilted lamella the first-acquired frame may be at a
significant tilt angle, but it is still the most stable and has the most
uniform illumination relative to the gain.  The first N acquisitions
(default 12) are used.

Output
------
  gain_image.png            — gain reference image (greyscale, for inspection)
  corrected_averages.png    — 2-row grid: raw×gain image + residual per transform
                              (4 rows if --include-divide is set)
  cv_vs_nmovies.png         — CV convergence per transform
                              (dashed ÷gain lines only with --include-divide)
  residual_progression.png  — best transform's residual at log-spaced checkpoints
  report.html               — standalone HTML viewer (no server required)
  aretomo3_project.json     — project state updated (backup copy written here)
"""

import re
import sys
import json
import random
import datetime
import numpy as np

from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Transform table
# AreTomo3 source (GFlip2D.cu, CLoadRefs.cpp):
#   FlipGain 1 → GFlip2D::Vertical()   → numpy flipud  (around horizontal axis)
#   FlipGain 2 → GFlip2D::Horizontal() → numpy fliplr  (around vertical axis)
#   RotGain  1 → 90° CCW, 2 → 180°, 3 → 270° CCW
#   Order applied: rotate → flip → invert
#
# Flag values here assume --software aretomo3 (gain pre-flipped with flipud
# before scoring, see run()).  With the pre-flip the mapping is 1:1 correct.
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    'none':   {'func': lambda g: g.copy(),              'rot_gain': 0, 'flip_gain': 0},
    'flipud': {'func': lambda g: np.flipud(g).copy(),   'rot_gain': 0, 'flip_gain': 1},
    'fliplr': {'func': lambda g: np.fliplr(g).copy(),   'rot_gain': 0, 'flip_gain': 2},
    'rot180': {'func': lambda g: np.rot90(g, 2).copy(), 'rot_gain': 2, 'flip_gain': 0},
}

_COLOURS = {
    'none':   '#90a4ae',
    'flipud': '#66bb6a',
    'fliplr': '#ef5350',
    'rot180': '#ffa726',
}

_MRC_MODE_NAMES = {
    0: 'int8',
    1: 'int16 (signed)',
    3: 'complex int16',
    4: 'complex float32',
    6: 'uint16',
}


# ─────────────────────────────────────────────────────────────────────────────
# Filename parsing
# Expected pattern: ..._001_14.00_20260213_171849_fractions.tiff
#                        ^^^  ^^^^^  acq   tilt
# ─────────────────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r'_(\d{3})_([-\d]+\.\d+)_\d{8}_\d{6}_fractions\.tiff?$',
    re.IGNORECASE,
)


def _parse_movie_name(path):
    """Return (acq_order: int, tilt_angle: float) from filename, or (None, None)."""
    m = _FNAME_RE.search(path.name)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


# ─────────────────────────────────────────────────────────────────────────────
# Gain loading and validation
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_validate_gain(gain_path):
    """
    Open gain MRC, validate mode (must be float32) and geometry (must be
    rectangular for K3 support).  Prints a formatted summary and exits on
    any failure.  Returns (gain_array: float32 ndarray, nx: int, ny: int).
    """
    try:
        import mrcfile
    except ImportError:
        print('ERROR: mrcfile is not installed.  Run: pip install mrcfile')
        sys.exit(1)

    print(f'Gain file:  {gain_path}')

    with mrcfile.open(str(gain_path), permissive=True) as mrc:
        mode = int(mrc.header.mode)
        nx   = int(mrc.header.nx)
        ny   = int(mrc.header.ny)
        data = mrc.data.copy()

    # 32-bit float check
    if mode != 2:
        mode_name = _MRC_MODE_NAMES.get(mode, f'unknown (mode {mode})')
        print(f'  Mode:     {mode_name}')
        print()
        print(f'ERROR: Gain file must be 32-bit float (MRC mode 2).')
        print(f'       Found: {mode_name}.')
        print(f'       Convert the gain to float32 before running this command.')
        sys.exit(1)

    print(f'  Mode:     float32 (mode 2)  \u2713')

    # Geometry check
    if nx == ny:
        print(f'  Size:     {nx} \u00d7 {ny}  (square)')
        print()
        print(f'ERROR: Square-sensor cameras (e.g., Falcon 3/4, {nx}\u00d7{ny}) '
              f'are not yet supported.')
        print(f'       For square sensors 8 transforms are valid '
              f'(4 rotations \u00d7 2 flip states);')
        print(f'       implementing this requires Falcon test data.')
        print(f'       Only rectangular K3 sensors are currently supported.')
        sys.exit(1)

    print(f'  Size:     {nx} \u00d7 {ny}  (rectangular \u2192 K3)  \u2713')

    if data.ndim == 3:
        data = data[0]
    return data.astype(np.float32), nx, ny


# ─────────────────────────────────────────────────────────────────────────────
# Square-camera stub (future)
# ─────────────────────────────────────────────────────────────────────────────

def _run_square_camera(gain, movies):
    """
    Placeholder for square-sensor cameras (e.g., Falcon 3/4, 4096 × 4096).

    For square sensors, 8 transforms are valid:
      4 rotations (0°, 90°, 180°, 270°) × 2 flip states (none, flipud)
    All 8 preserve the square dimensions, unlike the rectangular K3 case
    where only the 4 dimension-preserving transforms are valid.

    Implementation notes for when Falcon data is available:
      - EER is the primary movie format for Falcon cameras (not TIFF).
        EER rendering (dose fractionation, gain application) differs from
        TIFF and will need a separate reader.
      - AreTomo3 EER conventions may differ from MotionCor2/RELION — verify
        against AreTomo3 source before implementing.
      - The scoring approach (CV of corrected average) should transfer
        directly from the K3 implementation.

    Not yet implemented — no Falcon test data is available.
    """
    raise NotImplementedError(
        'Square-sensor camera (e.g., Falcon) gain transform search is not '
        'yet implemented.  Please open an issue if you have Falcon test data.'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Movie accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_steps(n, n_checkpoints=8):
    """Log-spaced checkpoint indices (1-based) biased toward early movies."""
    import math
    steps = set()
    for k in range(n_checkpoints):
        idx = int(round(math.exp(math.log(n) * (k + 1) / n_checkpoints)))
        steps.add(max(1, min(idx, n)))
    steps.add(n)
    return sorted(steps)


def _downsample(arr, factor=4):
    """Quick downsample by striding — no import needed."""
    return arr[::factor, ::factor].astype(np.float32)


def _accumulate(movies, transformed_gains, compute_div=False):
    """
    For each selected movie, sum sub-frames then accumulate:
      multiply: raw × gain  (AreTomo3 convention — always computed)
      divide:   raw ÷ gain  (only when compute_div=True)

    At log-spaced checkpoints the normalised residual (arr/mean - 1) is
    saved at 1/4 resolution for the progression plot.

    Parameters
    ----------
    movies : list of (Path, acq_order, tilt_angle)
    transformed_gains : dict  name -> float64 ndarray
    compute_div : bool  also accumulate raw÷gain (default False)

    Returns
    -------
    corr_mul      : dict  name -> float64 ndarray
    corr_div      : dict  name -> float64 ndarray  (values are None if compute_div=False)
    cv_hist_mul   : dict  name -> list of float
    cv_hist_div   : dict  name -> list of float     (empty lists if compute_div=False)
    checkpoints   : list of (n_movies_so_far,
                             {name: resid_downsampled_float32})
    """
    try:
        import tifffile
    except ImportError:
        print('ERROR: tifffile is not installed.  Run: pip install tifffile')
        sys.exit(1)

    names       = list(transformed_gains.keys())
    corr_mul    = {n: None for n in names}
    corr_div    = {n: None for n in names}
    cv_hist_mul = {n: [] for n in names}
    cv_hist_div = {n: [] for n in names}
    checkpoints = []
    ckpt_set    = set(_checkpoint_steps(len(movies)))

    for i, (path, acq, tilt) in enumerate(movies):
        movie   = tifffile.imread(str(path)).astype(np.float64)
        raw_sum = movie.sum(axis=0) if movie.ndim == 3 else movie

        for n in names:
            g = transformed_gains[n]

            # multiply: raw × gain (AreTomo3 convention)
            c = raw_sum * g
            if corr_mul[n] is None:
                corr_mul[n] = c.copy()
            else:
                corr_mul[n] += c
            m = corr_mul[n].mean()
            cv_hist_mul[n].append(corr_mul[n].std() / m if m > 0 else np.nan)

            # divide: raw ÷ gain (optional)
            if compute_div:
                safe_g = np.where(g > 0, g, np.nan)
                c = np.nan_to_num(raw_sum / safe_g, nan=0.0)
                if corr_div[n] is None:
                    corr_div[n] = c.copy()
                else:
                    corr_div[n] += c
                m = corr_div[n].mean()
                cv_hist_div[n].append(corr_div[n].std() / m if m > 0 else np.nan)

        # Save checkpoint (1-based count)
        step = i + 1
        if step in ckpt_set:
            snap = {}
            for n in names:
                arr  = corr_mul[n]
                resid = arr / arr.mean() - 1.0
                snap[n] = _downsample(resid)
            checkpoints.append((step, snap))

        print(f'  [{i+1:3d}/{len(movies)}]  acq {acq:03d}  tilt {tilt:+.2f}\u00b0',
              end='\r', flush=True)

    print()  # newline after progress line
    return corr_mul, corr_div, cv_hist_mul, cv_hist_div, checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score(corr_mul, corr_div):
    """
    Compute CV for each transform under multiply and divide modes.
    CV = std / mean of the accumulated corrected image — lower = flatter = better.
    Best transform selected by cv_mul (AreTomo3 multiplies the gain).
    """
    scores = {}
    for name in corr_mul:
        m_mul  = corr_mul[name].mean()
        cv_mul = float(corr_mul[name].std() / m_mul) if m_mul > 0 else np.nan
        cv_div = None
        if corr_div.get(name) is not None:
            m_div  = corr_div[name].mean()
            cv_div = float(corr_div[name].std() / m_div) if m_div > 0 else np.nan
        scores[name] = {'cv_mul': cv_mul, 'cv_div': cv_div}
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_corrected_averages(corr_mul, corr_div, scores, best, out_path,
                             include_div=False):
    """
    2 rows × 4 columns  (default, multiply-only):
      Row 0: raw × gain  — greyscale, mean-normalised
      Row 1: residual    — diverging RdBu_r (arr/mean - 1)

    4 rows × 4 columns  (with include_div=True):
      Row 0: raw × gain, Row 1: residual (×)
      Row 2: raw ÷ gain, Row 3: residual (÷)

    Correct transform → flat grey in residual rows.
    """
    names  = list(corr_mul.keys())
    ny, nx = corr_mul[names[0]].shape
    pan_w  = 5.0
    pan_h  = pan_w * ny / nx
    n_rows = 4 if include_div else 2
    fig_h  = pan_h * n_rows + 3.0
    fig, axes = plt.subplots(n_rows, 4, figsize=(pan_w * 4, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing consistent
    fig.patch.set_facecolor('#16213e')

    for col_i, name in enumerate(names):
        is_best = (name == best)
        col     = '#66bb6a' if is_best else '#e0e0e0'
        marker  = '  \u2190' if is_best else ''

        iter_pairs = [(corr_mul[name], 'cv_mul', 0, 1)]
        if include_div and corr_div.get(name) is not None:
            iter_pairs.append((corr_div[name], 'cv_div', 2, 3))

        for arr, cv_key, row_img, row_resid in iter_pairs:
            norm  = arr / arr.mean()
            resid = norm - 1.0
            lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
            vlim   = float(np.percentile(np.abs(resid), 99))

            # Image row
            ax = axes[row_img, col_i]
            ax.imshow(norm, cmap='gray', vmin=lo, vmax=hi,
                      aspect='equal', interpolation='nearest')
            cv_val = scores[name][cv_key]
            cv_str = f'{cv_val:.4f}' if cv_val is not None else 'n/a'
            title_col = col if row_img == 0 else '#e0e0e0'
            title_txt = (f'{name}{marker}  CV={cv_str}'
                         if row_img == 0 else f'CV={cv_str}')
            ax.set_title(title_txt, color=title_col, fontsize=10,
                         fontweight='bold' if (is_best and row_img == 0) else 'normal')
            ax.axis('off')

            # Residual row
            ax = axes[row_resid, col_i]
            ax.imshow(resid, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                      aspect='equal', interpolation='nearest')
            ax.set_title(f'residual  \u00b1{vlim:.3f}',
                         color='#e0e0e0', fontsize=9)
            ax.axis('off')

        if is_best:
            for row_i in range(n_rows):
                for spine in axes[row_i, col_i].spines.values():
                    spine.set_edgecolor('#66bb6a')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                axes[row_i, col_i].axis('on')
                axes[row_i, col_i].tick_params(
                    left=False, bottom=False,
                    labelleft=False, labelbottom=False)

    # Row labels on left
    if include_div:
        row_labels  = ['raw \u00d7 gain', 'residual (\u00d7)',
                       'raw \u00f7 gain', 'residual (\u00f7)']
        row_colours = ['#90caf9', '#ce93d8', '#ef9a9a', '#ffcc80']
    else:
        row_labels  = ['raw \u00d7 gain', 'residual']
        row_colours = ['#90caf9', '#ce93d8']

    for row_i, (label, rcol) in enumerate(zip(row_labels, row_colours)):
        axes[row_i, 0].set_ylabel(label, color=rcol, fontsize=10)
        axes[row_i, 0].axis('on')
        axes[row_i, 0].tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False)
        for spine in axes[row_i, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle('Gain-corrected averages — flat/grey residual = correct transform',
                 color='#90caf9', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)




def _plot_cv_convergence(cv_hist_mul, cv_hist_div, scores, best, out_path,
                         include_div=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#0d1b2a')

    for name in cv_hist_mul:
        lw  = 2.5 if name == best else 1.5
        col = _COLOURS.get(name, 'white')
        cv_mul = scores[name]['cv_mul']
        ax.plot(range(1, len(cv_hist_mul[name]) + 1), cv_hist_mul[name],
                label=f'{name} ×  (CV={cv_mul:.4f})',
                color=col, lw=lw, linestyle='-')
        if include_div and cv_hist_div.get(name):
            cv_div = scores[name]['cv_div']
            cv_div_s = f'{cv_div:.4f}' if cv_div is not None else 'n/a'
            ax.plot(range(1, len(cv_hist_div[name]) + 1), cv_hist_div[name],
                    label=f'{name} ÷  (CV={cv_div_s})',
                    color=col, lw=lw * 0.7, linestyle='--', alpha=0.6)

    ax.set_xlabel('Movies accumulated', color='#e0e0e0')
    ax.set_ylabel('CV  (std / mean)  — lower is flatter', color='#e0e0e0')
    title_suffix = ', dashed = raw÷gain' if include_div else ''
    ax.set_title(f'Flatness convergence — solid = raw×gain{title_suffix}',
                 color='#90caf9', fontsize=12)
    ax.tick_params(colors='#e0e0e0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#445')
    ax.legend(facecolor='#1e2a45', labelcolor='#e0e0e0', framealpha=0.8)
    ax.grid(True, alpha=0.2, color='#445')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_residual_progression(checkpoints, best, out_path):
    """
    Show how the residual (raw×gain / mean - 1) of the best transform
    evolves as more movies are accumulated.

    Columns = log-spaced checkpoints; single row.
    Diverging colormap: flat grey → correct gain transform.
    Color scale fixed to the final checkpoint so all panels are comparable.
    """
    if not checkpoints:
        return

    n_ckpt     = len(checkpoints)
    last_resid = checkpoints[-1][1][best]
    vlim       = float(np.percentile(np.abs(last_resid), 99))

    ny, nx = checkpoints[0][1][best].shape
    pan_w  = 3.0
    pan_h  = pan_w * ny / nx

    fig, axes = plt.subplots(1, n_ckpt, figsize=(pan_w * n_ckpt, pan_h + 1.2))
    if n_ckpt == 1:
        axes = [axes]
    fig.patch.set_facecolor('#16213e')

    for ax, (n_movies, snap) in zip(axes, checkpoints):
        resid = snap[best]
        ax.imshow(resid, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                  aspect='equal', interpolation='nearest')
        ax.set_title(f'n={n_movies}', color='#e0e0e0', fontsize=9)
        ax.axis('off')

    fig.suptitle(
        f'Residual progression — {best}  (flat/grey = correct, ±{vlim:.3f})',
        color='#90caf9', fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_gain_image(gain, gain_path, out_path):
    """
    Save a greyscale view of the gain reference for visual inspection.
    Colour scale clipped to 1st–99th percentile; stats annotated in title.
    """
    vlo = float(np.percentile(gain, 1))
    vhi = float(np.percentile(gain, 99))
    ny, nx = gain.shape
    fig_w = 14.0
    fig_h = fig_w * ny / nx

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.8))
    fig.patch.set_facecolor('#16213e')
    im = ax.imshow(gain, cmap='gray', vmin=vlo, vmax=vhi,
                   aspect='equal', interpolation='bilinear', origin='lower')
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Gain value', color='#e0e0e0')
    cb.ax.yaxis.set_tick_params(color='#e0e0e0', labelcolor='#e0e0e0')
    ax.axis('off')
    ax.set_title(
        f'{Path(gain_path).name}   {nx}\u00d7{ny}   '
        f'mean={gain.mean():.4f}   min={gain.min():.4f}   max={gain.max():.4f}',
        color='#90caf9', fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _make_standalone_html(results, out_path, include_div=False):
    best  = results['best_transform']
    flags = (f"-RotGain {results['aretomo3_rot_gain']} "
             f"-FlipGain {results['aretomo3_flip_gain']}")

    rows_html = ''
    for name, s in results['scores'].items():
        marker   = '  \u2190 best' if name == best else ''
        style    = 'color:#66bb6a;font-weight:bold' if name == best else ''
        cv_mul_s = f"{s['cv_mul']:.4f}" if s.get('cv_mul') is not None else 'n/a'
        row = (f'<tr style="{style}"><td>{name}{marker}</td>'
               f'<td>{cv_mul_s}</td>')
        if include_div:
            cv_div_s = f"{s['cv_div']:.4f}" if s.get('cv_div') is not None else 'n/a'
            row += f'<td>{cv_div_s}</td>'
        row += '</tr>\n'
        rows_html += row

    div_header = '<th>CV &#x2193; raw&#xf7;gain</th>' if include_div else ''

    tilt_lo, tilt_hi = results['tilt_range_deg']
    acq_lo,  acq_hi  = results['acq_range']

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Gain Transform Check</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', sans-serif;
      background: #16213e; color: #e0e0e0;
      padding: 30px 40px;
    }}
    h1 {{ color: #90caf9; margin-bottom: 6px; font-size: 1.3em; }}
    .sub {{ color: #78909c; font-size: 0.85em; margin-bottom: 24px; }}
    .card {{
      background: #1e2a45; border-radius: 10px; padding: 20px 24px;
      margin-bottom: 24px; max-width: 680px;
    }}
    .best {{ color: #66bb6a; font-size: 1.25em; font-weight: bold; }}
    .flags {{
      font-family: monospace; font-size: 1.05em; color: #ffcc80;
      margin-top: 8px; background: #0d1b2a; padding: 6px 12px;
      border-radius: 6px; display: inline-block; margin-top: 10px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ padding: 7px 14px; text-align: left; border-bottom: 1px solid #2e3f5c; }}
    th {{ color: #90caf9; font-weight: normal; }}
    .meta {{ color: #78909c; font-size: 0.82em; margin-top: 14px; }}
    .imgs {{
      display: flex; gap: 24px; flex-wrap: wrap; margin-top: 0;
    }}
    .imgs img {{
      max-width: 1400px; width: 100%; border-radius: 8px;
      border: 1px solid #2e3f5c;
    }}
  </style>
</head>
<body>
  <h1>Gain Transform Check</h1>
  <div class="sub">
    AreTomo3 &nbsp;|&nbsp; K3 TIFF movies + MRC gain &nbsp;|&nbsp;
    {results['timestamp'][:10]}
  </div>

  <div class="card">
    <div class="best">Best transform: {best}</div>
    <div class="flags">AreTomo3 flags: {flags}</div>
    <table>
      <tr>
        <th>Transform</th>
        <th>CV &#x2193; raw&#xd7;gain</th>
        {div_header}
      </tr>
      {rows_html}
    </table>
    <div class="meta">
      Gain: {results['gain_file']} &nbsp;|&nbsp;
      Acq &#x2264; {results['acq_order_threshold']} filter:
      {results['n_movies_after_filter']} movies &nbsp;|&nbsp;
      Sampled: {results['n_movies_tested']} &nbsp;|&nbsp;
      Acq range: {acq_lo:03d}&#x2013;{acq_hi:03d} &nbsp;|&nbsp;
      Tilt: {tilt_lo:+.2f}&#xb0; &#x2192; {tilt_hi:+.2f}&#xb0;
    </div>
  </div>

  <div class="imgs">
    <img src="gain_image.png" alt="Gain reference image">
    <img src="corrected_averages.png" alt="Corrected averages">
    <img src="cv_vs_nmovies.png" alt="CV convergence">
    <img src="residual_progression.png" alt="Residual progression vs movies accumulated">
  </div>
</body>
</html>
"""
    with open(out_path, 'w') as fh:
        fh.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'check-gain-transform',
        help='Determine AreTomo3 gain flip/rotation for K3 TIFF movies + MRC gain',
        formatter_class=__import__('argparse').ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--gain', '-g', required=True,
        help='Path to gain reference MRC file (must be float32, mode 2)',
    )
    p.add_argument(
        '--frames', '-f', required=True,
        help='Directory containing *_fractions.tiff movie files',
    )
    p.add_argument(
        '--output', '-o', default='gain_check',
        help='Output directory for PNGs, report.html, and project JSON backup',
    )
    p.add_argument(
        '--n-acquisitions', '-n', type=int, default=12,
        help='Maximum acquisition order to include (filter threshold). '
             'Only movies with acq_order <= this value are considered. '
             'Lower acquisitions = lower tilt = more uniform illumination.',
    )
    p.add_argument(
        '--n-movies', '-N', type=int, default=150,
        help='Number of movies to randomly sample from the filtered set '
             'and accumulate. 100–200 is sufficient for convergence.',
    )
    p.add_argument(
        '--include-divide', action='store_true', default=False,
        help='Also test raw÷gain for each transform and include it in the '
             'plots and report.  By default only raw×gain is computed '
             '(AreTomo3 convention).',
    )
    p.add_argument(
        '--software', default='aretomo3', choices=['aretomo3', 'none'],
        help='Target motion-correction software.  "aretomo3" (default) '
             'pre-applies flipud() to the gain before scoring to account for '
             'AreTomo3\'s implicit MRC Y-flip (MRC row 0 = bottom), so that '
             'reported flags map 1:1 to -RotGain/-FlipGain.  '
             '"none" disables the pre-flip (legacy, not recommended).',
    )
    p.set_defaults(func=run)
    return p


def run(args):
    gain_path  = Path(args.gain)
    frames_dir = Path(args.frames)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate gain ─────────────────────────────────────────────────────────
    gain, nx, ny = _load_and_validate_gain(gain_path)

    # AreTomo3 reads MRC gain with an implicit Y-flip (MRC row 0 = bottom).
    # Pre-flipping here means the TRANSFORMS flag values are 1:1 correct.
    software = getattr(args, 'software', 'aretomo3')
    if software == 'aretomo3':
        gain = np.flipud(gain).copy()
        print('  MRC Y-flip:  applied (--software aretomo3)')
        print('  Reported -RotGain/-FlipGain values are direct AreTomo3 flags.')
    else:
        print('  MRC Y-flip:  disabled (--software none)')
        print('  WARNING: reported flags may not map correctly to AreTomo3.')
    print()

    # ── Find and select movies ────────────────────────────────────────────────
    all_tiffs = sorted(frames_dir.glob('*_fractions.tiff'))
    if not all_tiffs:
        all_tiffs = sorted(frames_dir.glob('*_fractions.tif'))
    if not all_tiffs:
        print(f'ERROR: No *_fractions.tiff files found in {frames_dir}')
        sys.exit(1)

    parsed, n_unparseable = [], 0
    for p in all_tiffs:
        acq, tilt = _parse_movie_name(p)
        if acq is None:
            n_unparseable += 1
            continue
        parsed.append((p, acq, tilt))

    if n_unparseable:
        print(f'  WARNING: {n_unparseable} files did not match expected filename '
              f'pattern and were skipped.')
    if not parsed:
        print('ERROR: No movies with parseable acquisition order found.')
        sys.exit(1)

    # Filter to low-acquisition-order movies only
    filtered = [x for x in parsed if x[1] <= args.n_acquisitions]
    if not filtered:
        print(f'ERROR: No movies with acquisition order <= {args.n_acquisitions} found.')
        print(f'       Check --n-acquisitions or the filename pattern.')
        sys.exit(1)

    # Randomly sample up to --n-movies for accumulation (reproducible seed)
    if len(filtered) > args.n_movies:
        random.seed(42)
        selected = sorted(random.sample(filtered, args.n_movies), key=lambda x: x[1])
    else:
        selected = sorted(filtered, key=lambda x: x[1])

    acq_min  = min(x[1] for x in selected)
    acq_max  = max(x[1] for x in selected)
    tilt_min = min(x[2] for x in selected)
    tilt_max = max(x[2] for x in selected)

    print(f'Frames dir: {frames_dir}  ({len(all_tiffs)} movies found)')
    print(f'  After acq <= {args.n_acquisitions} filter: {len(filtered)} movies')
    print(f'  Sampling: {len(selected)} movies  '
          f'(acq range {acq_min:03d}\u2013{acq_max:03d}, '
          f'tilt range: {tilt_min:+.2f}\u00b0 \u2192 {tilt_max:+.2f}\u00b0)')
    print()

    # ── Build transformed gains (float64 for accumulation precision) ──────────
    transformed_gains = {
        name: t['func'](gain).astype(np.float64)
        for name, t in TRANSFORMS.items()
    }

    # ── Accumulate ────────────────────────────────────────────────────────────
    include_div = getattr(args, 'include_divide', False)
    n_tested = len(TRANSFORMS)
    div_note = ' (+ ÷gain)' if include_div else ''
    print(f'Testing {n_tested} transforms{div_note}...')
    corr_mul, corr_div, cv_hist_mul, cv_hist_div, checkpoints = _accumulate(
        selected, transformed_gains, compute_div=include_div)
    print()

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = _score(corr_mul, corr_div)
    best   = min(scores, key=lambda n: scores[n]['cv_mul'])

    if include_div:
        print(f'  {"Transform":<8}  {"CV×gain":>8}  {"CV÷gain":>8}')
    else:
        print(f'  {"Transform":<8}  {"CV×gain":>8}')
    for name, s in scores.items():
        marker = '  <- best' if name == best else ''
        if include_div:
            cv_div_s = f'{s["cv_div"]:8.4f}' if s['cv_div'] is not None else '     n/a'
            print(f'  {name:<8}  {s["cv_mul"]:>8.4f}  {cv_div_s}{marker}')
        else:
            print(f'  {name:<8}  {s["cv_mul"]:>8.4f}{marker}')
    print()

    rot_gain  = TRANSFORMS[best]['rot_gain']
    flip_gain = TRANSFORMS[best]['flip_gain']
    print(f'Best transform: {best}')
    print(f'AreTomo3 flags: -RotGain {rot_gain} -FlipGain {flip_gain}')
    print()

    # ── Build results dict for HTML ────────────────────────────────────────────
    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    results = {
        'gain_file':             str(gain_path),
        'frames_dir':            str(frames_dir),
        'input_type':            'tiff_mrc_aretomo3',
        'software':              software,
        'mrc_y_flip_applied':    software == 'aretomo3',
        'n_movies_after_filter': len(filtered),
        'n_movies_tested':       len(selected),
        'acq_order_threshold':   args.n_acquisitions,
        'acq_range':             [int(acq_min), int(acq_max)],
        'tilt_range_deg':        [float(tilt_min), float(tilt_max)],
        'best_transform':        best,
        'aretomo3_rot_gain':     rot_gain,
        'aretomo3_flip_gain':    flip_gain,
        'scores':                scores,
        'timestamp':             timestamp,
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    gain_img_path = out_dir / 'gain_image.png'
    avg_path      = out_dir / 'corrected_averages.png'
    cv_path       = out_dir / 'cv_vs_nmovies.png'
    prog_path     = out_dir / 'residual_progression.png'
    _plot_gain_image(gain, str(gain_path), str(gain_img_path))
    _plot_corrected_averages(corr_mul, corr_div, scores, best, str(avg_path),
                             include_div=include_div)
    _plot_cv_convergence(cv_hist_mul, cv_hist_div, scores, best, str(cv_path),
                         include_div=include_div)
    _plot_residual_progression(checkpoints, best, str(prog_path))

    # ── Standalone HTML ───────────────────────────────────────────────────────
    html_path = out_dir / 'report.html'
    _make_standalone_html(results, str(html_path), include_div=include_div)

    # ── Project JSON ──────────────────────────────────────────────────────────
    print()
    update_section(
        section    = 'gain_check',
        values     = {
            'command':               ' '.join(sys.argv),
            'args':                  args_to_dict(args),
            'timestamp':             timestamp,
            'best_transform':        best,
            'aretomo3_rot_gain':     rot_gain,
            'aretomo3_flip_gain':    flip_gain,
            'n_movies_after_filter': len(filtered),
            'n_movies_tested':       len(selected),
            'acq_order_threshold':   args.n_acquisitions,
            'acq_range':             [int(acq_min), int(acq_max)],
            'tilt_range_deg':        [float(tilt_min), float(tilt_max)],
            'scores':                scores,
            'output_dir':            str(out_dir),
        },
        backup_dir = out_dir,
    )

    proj_backup = out_dir / 'aretomo3_project.json'
    print()
    print('Output')
    print(f'  Gain image               : {gain_img_path}')
    avg_label = 'Corrected averages (×÷)' if include_div else 'Corrected averages (×)'
    print(f'  {avg_label:<24} : {avg_path}')
    print(f'  CV convergence           : {cv_path}')
    print(f'  Residual progression     : {prog_path}')
    print(f'  HTML report              : {html_path}')
    print(f'  Project backup           : {proj_backup}')
