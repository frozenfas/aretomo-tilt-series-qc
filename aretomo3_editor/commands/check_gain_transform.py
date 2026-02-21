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
    For each selected movie, sum sub-frames then accumulate two versions:
      multiply: raw × gain  (AreTomo3 convention)
      divide:   raw ÷ gain  (for comparison — should be worse)

    Parameters
    ----------
    movies : list of (Path, acq_order, tilt_angle)
    transformed_gains : dict  name -> float64 ndarray

    Returns
    -------
    corr_mul    : dict  name -> float64 ndarray  (accumulated raw × gain)
    corr_div    : dict  name -> float64 ndarray  (accumulated raw ÷ gain)
    cv_hist_mul : dict  name -> list of float    (CV after each movie, multiply)
    cv_hist_div : dict  name -> list of float    (CV after each movie, divide)
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

            # divide: raw ÷ gain (guard against zero/negative)
            safe_g = np.where(g > 0, g, np.nan)
            c = np.nan_to_num(raw_sum / safe_g, nan=0.0)
            if corr_div[n] is None:
                corr_div[n] = c.copy()
            else:
                corr_div[n] += c
            m = corr_div[n].mean()
            cv_hist_div[n].append(corr_div[n].std() / m if m > 0 else np.nan)

        print(f'  [{i+1:3d}/{len(movies)}]  acq {acq:03d}  tilt {tilt:+.2f}\u00b0',
              end='\r', flush=True)

    print()  # newline after progress line
    return corr_mul, corr_div, cv_hist_mul, cv_hist_div


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
        m_mul = corr_mul[name].mean()
        m_div = corr_div[name].mean()
        scores[name] = {
            'cv_mul': float(corr_mul[name].std() / m_mul) if m_mul > 0 else np.nan,
            'cv_div': float(corr_div[name].std() / m_div) if m_div > 0 else np.nan,
        }
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_corrected_averages(corr_mul, corr_div, scores, best, out_path):
    """
    Approach B: 3 rows × 4 columns.
    Row 0: raw × gain          — greyscale, mean-normalised
    Row 1: raw ÷ gain          — greyscale, mean-normalised
    Row 2: residual of row 0   — diverging colourmap (arr/mean - 1);
           deviations from flat show as red/blue; correct transform → grey
    """
    names  = list(corr_mul.keys())
    ny, nx = corr_mul[names[0]].shape
    pan_w  = 5.0
    pan_h  = pan_w * ny / nx
    fig_h  = pan_h * 3 + 2.5
    fig, axes = plt.subplots(3, 4, figsize=(pan_w * 4, fig_h))
    fig.patch.set_facecolor('#16213e')

    for col_i, name in enumerate(names):
        is_best = (name == best)
        col     = '#66bb6a' if is_best else '#e0e0e0'
        marker  = '  \u2190' if is_best else ''

        # Row 0: raw × gain
        ax   = axes[0, col_i]
        norm = corr_mul[name] / corr_mul[name].mean()
        lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
        ax.imshow(norm, cmap='gray', vmin=lo, vmax=hi,
                  aspect='equal', interpolation='nearest')
        ax.set_title(f'{name}{marker}  CV={scores[name]["cv_mul"]:.4f}',
                     color=col, fontsize=10,
                     fontweight='bold' if is_best else 'normal')
        ax.axis('off')

        # Row 1: raw ÷ gain
        ax   = axes[1, col_i]
        norm = corr_div[name] / corr_div[name].mean()
        lo, hi = np.percentile(norm, 1), np.percentile(norm, 99)
        ax.imshow(norm, cmap='gray', vmin=lo, vmax=hi,
                  aspect='equal', interpolation='nearest')
        ax.set_title(f'CV={scores[name]["cv_div"]:.4f}',
                     color='#e0e0e0', fontsize=9)
        ax.axis('off')

        # Row 2: residual (raw × gain) — diverging, symmetric around 0
        ax     = axes[2, col_i]
        resid  = corr_mul[name] / corr_mul[name].mean() - 1.0
        vlim   = float(np.percentile(np.abs(resid), 99))
        im     = ax.imshow(resid, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                           aspect='equal', interpolation='nearest')
        ax.set_title(f'residual  (±{vlim:.3f})',
                     color='#e0e0e0', fontsize=9)
        ax.axis('off')
        if is_best:
            for row_i in range(3):
                for spine in axes[row_i, col_i].spines.values():
                    spine.set_edgecolor('#66bb6a')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
                axes[row_i, col_i].axis('on')
                axes[row_i, col_i].tick_params(
                    left=False, bottom=False,
                    labelleft=False, labelbottom=False)

    # Row labels
    row_labels = ['raw × gain', 'raw ÷ gain', 'residual (×gain)']
    row_colours = ['#90caf9', '#ef9a9a', '#ce93d8']
    for row_i, (label, rcol) in enumerate(zip(row_labels, row_colours)):
        axes[row_i, 0].set_ylabel(label, color=rcol, fontsize=10)
        axes[row_i, 0].axis('on')
        axes[row_i, 0].tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False)
        for spine in axes[row_i, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle('Gain-corrected averages — flat image = correct transform\n'
                 'Row 3: residual from flat (red/blue = non-uniformity)',
                 color='#90caf9', fontsize=13)
    plt.tight_layout()
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
        marker   = '  \u2190 best' if name == best else ''
        style    = 'color:#66bb6a;font-weight:bold' if name == best else ''
        cv_mul_s = f"{s['cv_mul']:.4f}" if s.get('cv_mul') is not None else 'n/a'
        cv_div_s = f"{s['cv_div']:.4f}" if s.get('cv_div') is not None else 'n/a'
        rows_html += (
            f'<tr style="{style}"><td>{name}{marker}</td>'
            f'<td>{cv_mul_s}</td><td>{cv_div_s}</td></tr>\n'
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
    <img src="corrected_averages.png" alt="Corrected averages (multiply vs divide)">
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
    corr_mul, corr_div, cv_hist_mul, cv_hist_div = _accumulate(
        selected, transformed_gains)
    print()

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = _score(corr_mul, corr_div)
    best   = min(scores, key=lambda n: scores[n]['cv_mul'])

    print(f'  {"Transform":<8}  {"CV×gain":>8}  {"CV÷gain":>8}')
    for name, s in scores.items():
        marker = '  <- best' if name == best else ''
        print(f'  {name:<8}  {s["cv_mul"]:>8.4f}  {s["cv_div"]:>8.4f}{marker}')
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
    cv_path  = out_dir / 'cv_vs_nmovies.png'
    _plot_corrected_averages(corr_mul, corr_div, scores, best, str(avg_path))
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
    print(f'  Corrected averages (×÷): {avg_path}')
    print(f'  CV convergence           : {cv_path}')
    print(f'  HTML report              : {html_path}')
    print(f'  Project backup           : {proj_backup}')
