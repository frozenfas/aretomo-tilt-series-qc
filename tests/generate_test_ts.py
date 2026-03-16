#!/usr/bin/env python3
"""
generate_test_ts.py — synthetic tilt series for gain-correction testing.

Creates a tilt series where the 'true signal' at each tilt is the Cameraman
test image (skimage.data.camera, 512×512, public domain) multiplied by
cos(tilt_angle).  A known asymmetric diagonal gain is baked in:

    raw[y, x] = true_signal[y, x] / gain[y, x]  + Poisson noise + drift

The correct AreTomo3 command applies:  corrected = raw × gain  →  true_signal.
Any wrong gain orientation leaves a visible diagonal-gradient artefact.

Gain design
-----------
    gain[y, x] = 0.5 + 0.3 * x / (W-1) + 0.7 * y / (H-1)   (range 0.5 – 1.5)

    top-left  ≈ 0.50 (dark)    top-right  ≈ 0.80
    bot-left  ≈ 1.20           bot-right  ≈ 1.50 (bright)

    Unequal x/y slopes (0.3 vs 0.7) ensure all 8 RotGain × FlipGain
    combinations are distinguishable, even for square images.

Usage
-----
    python tests/generate_test_ts.py --output synth_ts
    python tests/generate_test_ts.py --self-test   # quick numerical check (no files)
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys
from pathlib import Path

import numpy as np

try:
    import mrcfile
except ImportError:
    sys.exit('ERROR: mrcfile not installed.  conda install -c conda-forge mrcfile')

try:
    import tifffile
except ImportError:
    sys.exit('ERROR: tifffile not installed.   conda install -c conda-forge tifffile')

try:
    from skimage.data import camera as _camera
except ImportError:
    sys.exit('ERROR: scikit-image not installed.  conda install scikit-image')


# ── physical constants ────────────────────────────────────────────────────────
PIXEL_SPACING = 10.0    # Å/px  (coarse → AreTomo3 runs quickly)
DOSE_PER_TILT =  4.0    # e⁻/Å²  total per tilt
N_SUBFRAMES   =  4      # sub-frames per tilt movie
SIGNAL_SCALE  = 400.0   # scale cameraman → mean raw counts at 0° tilt
DARK_LEVEL    =  30.0   # dark current counts
DRIFT_MAX_PX  =  1.5    # max random integer shift per sub-frame (px)


# ── gain helpers ──────────────────────────────────────────────────────────────

def _make_gain(H: int, W: int) -> np.ndarray:
    """
    Asymmetric diagonal gradient gain, float32, range [0.5, 1.5].

    Uses *different* slopes for x and y (0.3 vs 0.7) so that all 8
    combinations of RotGain (0/1/2/3) × FlipGain (0/1) are distinct even
    for square images (equal slopes create a symmetry where rot90+flipud ≡
    identity).
    """
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    return 0.5 + 0.3 * xx + 0.7 * yy


def _transform_gain(gain: np.ndarray, rot: int, flip: int) -> np.ndarray:
    """
    Apply AreTomo3 -RotGain / -FlipGain to a gain array.

    rot  = 0 (none), 1 (90° CCW), 2 (180°), 3 (270° CCW)
    flip = 0 (none), 1 (flipud / flip Y)
    """
    g = np.rot90(gain, k=rot)
    if flip:
        g = np.flipud(g)
    return g


# ── numerical self-test ───────────────────────────────────────────────────────

def _run_self_test() -> None:
    """
    Numerical self-test (no AreTomo3 required).

    For each RotGain/FlipGain combination, apply the transformed gain to one
    raw image and measure how well the true signal is recovered.
    Expected result: RotGain=0 FlipGain=0 (identity) is best.
    """
    print('Self-test: numerical recovery check for all gain orientations')
    print('─' * 72)

    rng    = np.random.default_rng(42)
    cam    = _camera().astype(np.float32)
    H, W   = cam.shape
    gain   = _make_gain(H, W)
    signal = cam / cam.mean() * SIGNAL_SCALE   # true signal at 0° tilt

    # Generate one noisy raw image (0° tilt, no drift for clarity)
    raw_ideal = signal / gain
    lam  = (raw_ideal + DARK_LEVEL).clip(min=0)
    raw  = rng.poisson(lam).astype(np.float32)

    orientations = [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 0), (3, 1),
    ]

    results = []
    for rot, flip in orientations:
        applied_gain = _transform_gain(gain, rot, flip)
        corrected    = raw * applied_gain

        # Pearson r with true signal
        r = float(np.corrcoef(corrected.ravel(), signal.ravel())[0, 1])

        # CV of corrected/signal ratio (0 = perfect flat ratio → perfect correction)
        ratio    = corrected / signal.clip(min=1.0)
        ratio_cv = float(ratio.std() / ratio.mean())

        results.append((rot, flip, r, ratio_cv))

    results.sort(key=lambda x: -x[2])   # sort by correlation, best first

    hdr = f"{'RotGain':>8}  {'FlipGain':>8}  {'corr(corrected,signal)':>24}  {'CV(corrected/signal)':>22}"
    print(hdr)
    print(f"{'':>8}  {'':>8}  {'[higher = better]':>24}  {'[lower = better]':>22}")
    print('─' * 68)
    for rot, flip, r, cv in results:
        marker = '  ← BEST' if (rot, flip) == (results[0][0], results[0][1]) else ''
        print(f'{rot:>8}  {flip:>8}  {r:>24.6f}  {cv:>22.6f}{marker}')

    print()
    best_rot, best_flip = results[0][0], results[0][1]
    if best_rot == 0 and best_flip == 0:
        print('PASS — identity transform (RotGain=0 FlipGain=0) gives best recovery.')
    else:
        print(f'FAIL — best orientation is RotGain={best_rot} FlipGain={best_flip}.')
        print('       This may indicate a bug in _make_gain or _transform_gain.')
    print()


# ── mdoc writer ───────────────────────────────────────────────────────────────

def _write_mdoc(
    path: Path,
    tilt_angles: np.ndarray,
    movie_paths: list[Path],
    H: int,
    W: int,
) -> None:
    t0 = datetime.datetime(2024, 1, 15, 10, 0, 0)
    with open(path, 'w') as fh:
        fh.write('[T = Synthetic tilt series - cameraman gain test]\n\n')
        fh.write(f'PixelSpacing = {PIXEL_SPACING}\n')
        fh.write(f'ImageFile = ts-001.mrc\n')
        fh.write(f'ImageSize = {W} {H}\n')
        fh.write('DataMode = 6\n\n')
        for z, (tilt, mp) in enumerate(zip(tilt_angles, movie_paths)):
            dt = t0 + datetime.timedelta(seconds=z * 90)
            fh.write(f'[ZValue = {z}]\n')
            fh.write(f'TiltAngle = {tilt:.2f}\n')
            fh.write(f'ExposureDose = {DOSE_PER_TILT:.2f}\n')
            fh.write(f'PixelSpacing = {PIXEL_SPACING}\n')
            fh.write(f'SubFramePath = {mp.resolve()}\n')
            fh.write(f'DateTime = {dt.strftime("%d-%b-%Y %H:%M:%S")}\n')
            fh.write('StageX = 0.0\nStageY = 0.0\nStageZ = 0.0\n')
            fh.write('Defocus = -3.0\n\n')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--output', '-o', default='synth_ts',
                    help='Output directory (default: synth_ts)')
    ap.add_argument('--n-tilts', type=int, default=41,
                    help='Number of tilts (default: 41, -60° to +60°)')
    ap.add_argument('--n-subframes', type=int, default=N_SUBFRAMES,
                    help=f'Sub-frames per tilt movie (default: {N_SUBFRAMES})')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
    ap.add_argument('--self-test', action='store_true',
                    help='Run numerical self-test and exit (no files written)')
    args = ap.parse_args()

    if args.self_test:
        _run_self_test()
        return

    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── image & gain ──────────────────────────────────────────────────────────
    cam    = _camera().astype(np.float32)   # 512 × 512, range [0, 255]
    H, W   = cam.shape
    gain   = _make_gain(H, W)               # (H, W) float32, range [0.5, 1.5]
    signal = cam / cam.mean() * SIGNAL_SCALE

    tilt_angles = np.linspace(-60.0, 60.0, args.n_tilts)

    # ── generate TIFF movies ──────────────────────────────────────────────────
    print(f'Generating {args.n_tilts} TIFF movies ({args.n_subframes} sub-frame(s) each) …')
    movie_paths: list[Path] = []

    for z, tilt in enumerate(tilt_angles):
        cos_t     = max(abs(math.cos(math.radians(tilt))), 0.05)
        raw_ideal = signal * cos_t / gain   # what AreTomo3 receives as raw input

        frames = []
        for _ in range(args.n_subframes):
            # Random integer drift per sub-frame
            dy = int(round(rng.uniform(-DRIFT_MAX_PX, DRIFT_MAX_PX)))
            dx = int(round(rng.uniform(-DRIFT_MAX_PX, DRIFT_MAX_PX)))
            shifted = np.roll(np.roll(raw_ideal, dy, axis=0), dx, axis=1)
            lam     = (shifted + DARK_LEVEL).clip(min=0)
            frames.append(rng.poisson(lam).astype(np.uint16))

        movie_arr  = np.stack(frames, axis=0)   # (N_SUBFRAMES, H, W) uint16
        fname      = f'ts-001_{z + 1:04d}.tiff'
        movie_path = out_dir / fname
        tifffile.imwrite(str(movie_path), movie_arr)
        movie_paths.append(movie_path)

    # ── save gain.mrc ─────────────────────────────────────────────────────────
    gain_path = out_dir / 'gain.mrc'
    print(f'Saving  {gain_path} …')
    with mrcfile.new(str(gain_path), overwrite=True) as mrc:
        mrc.set_data(gain)
        mrc.voxel_size = PIXEL_SPACING

    # ── write mdoc ────────────────────────────────────────────────────────────
    mdoc_path = out_dir / 'ts-001.mdoc'
    print(f'Writing {mdoc_path} …')
    _write_mdoc(mdoc_path, tilt_angles, movie_paths, H, W)

    # ── print AreTomo3 commands ───────────────────────────────────────────────
    SEP = '─' * 72
    print(f'\n{SEP}')
    print('Gain baked into raw data:')
    print(f'  gain[y,x] = 0.5 + 0.3*x/{W-1} + 0.7*y/{H-1}   (range 0.5 → 1.5)')
    print('  top-left  ≈ 0.50 (dark)    top-right  ≈ 0.80')
    print('  bot-left  ≈ 1.20           bot-right  ≈ 1.50 (bright)')
    print(f'{SEP}')
    print()
    print('AreTomo3 test commands — Cmd=0 (motion correction only).')
    print('  CORRECT orientation → cameraman visible in every tilt image.')
    print('  Wrong orientation   → diagonal gradient artefact.\n')

    orientations = [
        (0, 0, 'CORRECT — identity (no transform)'),
        (0, 1, 'wrong   — flipud'),
        (2, 0, 'wrong   — rot180'),
        (2, 1, 'wrong   — fliplr  (rot180 + flipud)'),
        (1, 0, 'wrong   — rot90 CCW'),
        (3, 0, 'wrong   — rot270 CCW'),
        (1, 1, 'wrong   — rot90 CCW + flipud'),
        (3, 1, 'wrong   — rot270 CCW + flipud'),
    ]

    # Trailing slash is required: AreTomo3 builds the movie path as
    # m_acInDir + filename_from_mdoc, so InPrefix must end with '/'.
    in_prefix = str(out_dir.resolve()) + '/'

    for rot, flip, label in orientations:
        run_dir = out_dir / f'run_rot{rot}_flip{flip}'
        print(f'# {label}')
        print(
            f'/opt/AreTomo3/AreTomo3'
            f' -InMdoc {mdoc_path.resolve()}'
            f' -InPrefix {in_prefix}'
            f' -OutDir {run_dir.resolve()}'
            f' -Gain {gain_path.resolve()}'
            f' -RotGain {rot} -FlipGain {flip}'
            f' -PixSize {PIXEL_SPACING}'
            f' -Kv 300 -Cs 2.7'
            f' -Gpu 3'
            f' -Cmd 0\n'
        )

    print(f'{SEP}')
    print()
    print('Self-test (no AreTomo3 required — verifies gain math in Python):')
    print(f'  python {Path(__file__).name} --self-test')
    print()
    print('Inspect tilt averages after AreTomo3 (requires IMOD):')
    print('  3dmod  <OutDir>/ts-001.mrc')
    print()
    print(f'Files written to: {out_dir.resolve()}/')
    print(f'{SEP}')


if __name__ == '__main__':
    main()
