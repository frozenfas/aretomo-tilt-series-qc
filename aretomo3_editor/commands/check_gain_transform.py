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
  corrected_averages.png — 2×2 grid of normalised corrected images
  cv_vs_nmovies.png     — CV convergence per transform vs movies accumulated
  report.html           — standalone HTML viewer (no server required)
  aretomo3_project.json  — project state updated (backup copy written here)
"""

import re
import sys
import json
import random
import datetime
import numpy as np

from aretomo3_editor.shared.project_json import update_section, args_to_dict
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

def _accumulate(movies, transformed_gains):
    """
    For each selected movie, sum sub-frames then:
      (A) accumulate the raw sum (no gain applied)     — Approach A
      (B×) multiply by each transformed gain           — Approach B multiply
      (B÷) divide by each transformed gain             — Approach B divide

    Parameters
    ----------
    movies : list of (Path, acq_order, tilt_angle)
    transformed_gains : dict  name -> float64 ndarray

    Returns
    -------
    corr_mul     : dict  name -> float64 ndarray  (accumulated raw × gain)
    corr_div     : dict  name -> float64 ndarray  (accumulated raw ÷ gain)
    raw_accum    : float64 ndarray                (accumulated raw, no gain)
    cv_hist_mul  : dict  name -> list of float    (CV after each movie, multiply)
    cv_hist_div  : dict  name -> list of float    (CV after each movie, divide)
    """
    try:
        import tifffile
    except ImportError:
        print('ERROR: tifffile is not installed.  Run: pip install tifffile')
        sys.exit(1)

    names       = list(transformed_gains.keys())
    corr_mul    = {n: None for n in names}
    corr_div    = {n: None for n in names}
    raw_accum   = None
    cv_hist_mul = {n: [] for n in names}
    cv_hist_div = {n: [] for n in names}

    for i, (path, acq, tilt) in enumerate(movies):
        movie   = tifffile.imread(str(path)).astype(np.float64)
        raw_sum = movie.sum(axis=0) if movie.ndim == 3 else movie

        # Approach A: accumulate raw (no gain)
        if raw_accum is None:
            raw_accum = raw_sum.copy()
        else:
            raw_accum += raw_sum

        for n in names:
            g = transformed_gains[n]

            # Approach B multiply: raw × gain (AreTomo3 convention)
            c = raw_sum * g
            if corr_mul[n] is None:
                corr_mul[n] = c.copy()
            else:
                corr_mul[n] += c
            m = corr_mul[n].mean()
            cv_hist_mul[n].append(corr_mul[n].std() / m if m > 0 else np.nan)

            # Approach B divide: raw ÷ gain (guard against zero/negative)
            safe_g = np.where(g > 0, g, np.nan)
            c = raw_sum / safe_g
            c = np.nan_to_num(c, nan=0.0)
            if corr_div[n] is None:
                corr_div[n] = c.copy()
            else:
                corr_div[n] += c
            m = corr_div[n].mean()
            cv_hist_div[n].append(corr_div[n].std() / m if m > 0 else np.nan)

        print(f'  [{i+1:3d}/{len(movies)}]  acq {acq:03d}  tilt {tilt:+.2f}\u00b0',
              end='\r', flush=True)

    print()  # newline after progress line
    return corr_mul, corr_div, raw_accum, cv_hist_mul, cv_hist_div


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _norm01(arr):
    """Normalise array to [0, 1] range."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr, dtype=np.float32)


def _cv(arr):
    m = arr.mean()
    return float(arr.std() / m) if m > 0 else np.nan


def _ssim_vs_flat(arr, ssim_fn):
    m = arr.mean()
    norm = (arr / m).astype(np.float32)
    dr   = float(norm.max() - norm.min())
    return float(ssim_fn(norm, np.ones_like(norm), data_range=dr)) if dr > 0 else 1.0


def _score(corr_mul, corr_div, raw_accum, transformed_gains):
    """
    Compute metrics for each transform under two gain application modes.

    Approach A — raw average vs gain orientations:
      ssim_vs_raw  : SSIM between the normalised raw average and each
                     transformed gain image.  Higher → patterns match → better.

    Approach B — corrected average vs flat:
      cv_mul / cv_div           : CV of (raw × gain) or (raw ÷ gain) accumulation.
                                  Lower → flatter → better.
      ssim_vs_flat_mul / _div   : SSIM vs uniform reference.  Higher → better.

    Best transform is selected by cv_mul (AreTomo3 multiplies the gain).
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        _has_skimage = True
    except ImportError:
        _has_skimage = False

    scores = {}
    for name in corr_mul:
        # Approach B: multiply
        cv_mul        = _cv(corr_mul[name])
        ssim_flat_mul = _ssim_vs_flat(corr_mul[name], ssim_fn) if _has_skimage else None

        # Approach B: divide
        cv_div        = _cv(corr_div[name])
        ssim_flat_div = _ssim_vs_flat(corr_div[name], ssim_fn) if _has_skimage else None

        # Approach A: raw average vs gain transform
        ssim_vs_raw = None
        if _has_skimage and raw_accum is not None:
            a = _norm01(raw_accum).astype(np.float32)
            b = _norm01(transformed_gains[name]).astype(np.float32)
            ssim_vs_raw = float(ssim_fn(a, b, data_range=1.0))

        scores[name] = {
            'cv_mul':           cv_mul,
            'cv_div':           cv_div,
            'ssim_vs_flat_mul': ssim_flat_mul,
            'ssim_vs_flat_div': ssim_flat_div,
            'ssim_vs_raw':      ssim_vs_raw,
        }

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_corrected_averages(corr_mul, corr_div, scores, best, out_path):
    """
    Approach B: 2 rows × 4 columns.
    Top row    = raw × gain  (AreTomo3 convention)
    Bottom row = raw ÷ gain  (inverse, for comparison)
    """
    names = list(corr_mul.keys())
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('#16213e')

    for col_i, name in enumerate(names):
        is_best = (name == best)
        col     = '#66bb6a' if is_best else '#e0e0e0'
        marker  = '  \u2190' if is_best else ''

        for row_i, (arr, cv_key) in enumerate([
            (corr_mul[name], 'cv_mul'),
            (corr_div[name], 'cv_div'),
        ]):
            ax   = axes[row_i, col_i]
            norm = arr / arr.mean()
            lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
            ax.imshow(norm, cmap='gray', vmin=lo, vmax=hi,
                      aspect='auto', interpolation='nearest')
            cv_s = f"CV={scores[name][cv_key]:.4f}"
            ax.set_title(f'{name}{marker}  {cv_s}', color=col, fontsize=10,
                         fontweight='bold' if is_best else 'normal')
            ax.axis('off')
            if is_best:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#66bb6a')
                    spine.set_linewidth(3)
                    spine.set_visible(True)

    # Row labels on the left
    axes[0, 0].set_ylabel('raw × gain', color='#90caf9', fontsize=11)
    axes[1, 0].set_ylabel('raw ÷ gain', color='#ef9a9a', fontsize=11)
    for ax in axes[:, 0]:
        ax.axis('on')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Approach B — gain-corrected averages (flat = correct transform)',
                 color='#90caf9', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_raw_vs_gains(raw_accum, n_movies, transformed_gains, scores, best, out_path):
    """
    Approach A: raw average (left, large) alongside the 4 candidate gain
    orientations (right, 2×2).  The correct gain orientation should visually
    match the raw average pattern.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor('#16213e')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.12)

    # ── Left: raw average ─────────────────────────────────────────────────────
    ax_raw  = fig.add_subplot(gs[:, 0])
    raw_avg = raw_accum / n_movies
    lo, hi  = np.percentile(raw_avg, 1), np.percentile(raw_avg, 99)
    ax_raw.imshow(raw_avg, cmap='gray', vmin=lo, vmax=hi,
                  aspect='auto', interpolation='nearest')
    ax_raw.set_title(f'Raw average\n({n_movies} movies, no gain applied)',
                     color='#90caf9', fontsize=11)
    ax_raw.axis('off')

    # ── Right: 2×2 gain transforms ────────────────────────────────────────────
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for (r, c), name in zip(positions, transformed_gains):
        ax      = fig.add_subplot(gs[r, c])
        gain_t  = transformed_gains[name]
        lo, hi  = np.percentile(gain_t, 1), np.percentile(gain_t, 99)
        ax.imshow(gain_t, cmap='gray', vmin=lo, vmax=hi,
                  aspect='auto', interpolation='nearest')

        is_best   = (name == best)
        col       = '#66bb6a' if is_best else '#e0e0e0'
        marker    = '  \u2190 best' if is_best else ''
        ssim_s    = (f"{scores[name]['ssim_vs_raw']:.4f}"
                     if scores[name]['ssim_vs_raw'] is not None else 'n/a')
        ax.set_title(f'{name}{marker}\nSSIM vs raw = {ssim_s}',
                     color=col, fontsize=10,
                     fontweight='bold' if is_best else 'normal')
        ax.axis('off')
        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor('#66bb6a')
                spine.set_linewidth(3)
                spine.set_visible(True)

    fig.suptitle('Approach A — raw average vs candidate gain orientations\n'
                 '(correct orientation should match the raw average pattern)',
                 color='#90caf9', fontsize=13)
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_cv_convergence(cv_hist_mul, scores, best, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#0d1b2a')

    for name, vals in cv_hist_mul.items():
        lw    = 2.5 if name == best else 1.5
        col   = _COLOURS.get(name, 'white')
        label = f'{name}  (final CV = {scores[name]["cv_mul"]:.4f})'
        ax.plot(range(1, len(vals) + 1), vals, label=label, color=col, lw=lw)

    ax.set_xlabel('Movies accumulated', color='#e0e0e0')
    ax.set_ylabel('CV  (std / mean)  — lower is flatter', color='#e0e0e0')
    ax.set_title('Approach B (raw × gain) — flatness convergence by transform',
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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _make_standalone_html(results, out_path):
    best  = results['best_transform']
    flags = (f"-RotGain {results['aretomo3_rot_gain']} "
             f"-FlipGain {results['aretomo3_flip_gain']}")

    rows_html = ''
    for name, s in results['scores'].items():
        marker        = '  \u2190 best' if name == best else ''
        style         = 'color:#66bb6a;font-weight:bold' if name == best else ''
        cv_mul_s      = f"{s['cv_mul']:.4f}"          if s.get('cv_mul')           is not None else 'n/a'
        cv_div_s      = f"{s['cv_div']:.4f}"          if s.get('cv_div')           is not None else 'n/a'
        ssim_raw_s    = f"{s['ssim_vs_raw']:.4f}"     if s.get('ssim_vs_raw')      is not None else 'n/a'
        rows_html += (
            f'<tr style="{style}"><td>{name}{marker}</td>'
            f'<td>{cv_mul_s}</td><td>{cv_div_s}</td><td>{ssim_raw_s}</td></tr>\n'
        )

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
      max-width: 720px; width: 100%; border-radius: 8px;
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
        <th>CV &#x2193; raw&#xf7;gain</th>
        <th>SSIM raw vs gain &#x2191;</th>
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
    <img src="raw_vs_gains.png" alt="Approach A: raw average vs gain orientations">
    <img src="corrected_averages.png" alt="Approach B: corrected averages (multiply vs divide)">
    <img src="cv_vs_nmovies.png" alt="CV convergence">
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
    p.set_defaults(func=run)
    return p


def run(args):
    gain_path  = Path(args.gain)
    frames_dir = Path(args.frames)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate gain ─────────────────────────────────────────────────────────
    gain, nx, ny = _load_and_validate_gain(gain_path)
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
    print('Testing 4 transforms...')
    corr_mul, corr_div, raw_accum, cv_hist_mul, cv_hist_div = _accumulate(
        selected, transformed_gains)
    print()

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = _score(corr_mul, corr_div, raw_accum, transformed_gains)
    best   = min(scores, key=lambda n: scores[n]['cv_mul'])

    print(f'  {"Transform":<8}  {"CV×gain":>8}  {"CV÷gain":>8}  {"SSIM-raw":>10}')
    for name, s in scores.items():
        marker       = '  <- best' if name == best else ''
        ssim_raw_str = f"{s['ssim_vs_raw']:.4f}" if s['ssim_vs_raw'] is not None else '       n/a'
        print(f'  {name:<8}  {s["cv_mul"]:>8.4f}  {s["cv_div"]:>8.4f}  '
              f'{ssim_raw_str:>10}{marker}')
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
    avg_path = out_dir / 'corrected_averages.png'
    raw_path = out_dir / 'raw_vs_gains.png'
    cv_path  = out_dir / 'cv_vs_nmovies.png'
    _plot_corrected_averages(corr_mul, corr_div, scores, best, str(avg_path))
    _plot_raw_vs_gains(raw_accum, len(selected), transformed_gains,
                       scores, best, str(raw_path))
    _plot_cv_convergence(cv_hist_mul, scores, best, str(cv_path))

    # ── Standalone HTML ───────────────────────────────────────────────────────
    html_path = out_dir / 'report.html'
    _make_standalone_html(results, str(html_path))

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
    print(f'  Approach A (raw vs gains): {raw_path}')
    print(f'  Approach B (×÷ corrected): {avg_path}')
    print(f'  CV convergence           : {cv_path}')
    print(f'  HTML report              : {html_path}')
    print(f'  Project backup           : {proj_backup}')
