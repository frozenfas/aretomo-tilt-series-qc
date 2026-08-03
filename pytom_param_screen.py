#!/usr/bin/env python3
"""
pytom_param_screen.py -- sparse parameter screen for ribosome picking.

Utility script kept in the repo for reuse, but NOT a maintained CLI
subcommand and NOT general-purpose -- every fixed value below (PROJECT_DIR,
TS_SELECT_CSV, GPUS, MIRROR, the MATRIX cells, VOLUME_SPLIT_BY_APIX) is
hardcoded for one specific screen (bi38262-30-akinetes, 2026-08-01). Treat
it as a template: copy/edit the "Fixed config" block below for a new
project/screen rather than expecting to run this file as-is against
different data.

Matching + extraction is done via a DIRECT in-process call to
pytom_match.run(), constructing an argparse.Namespace by hand -- the same
pattern pytom_ribo_auto.py's own --check-handedness already uses, NOT by
shelling out to the `aretomo3-preprocess pytom-match` CLI. This is
required, not just a style choice: the CLI's own argparse makes
--particle-diameter and --angular-search mutually exclusive (one flag
group), but pytom_extract_candidates.py *requires* a particle diameter to
mask around each peak during extraction regardless of how the search's
angular resolution was chosen -- so a real run needs both set
simultaneously, which only a directly-built Namespace can do (skips
argparse validation entirely, exactly like pytom_ribo_auto.py's own
_matching_defaults() + particle_diameter combination). The actual GPU
work (pytom_match_template.py / pytom_extract_candidates.py) still runs
as real subprocess.run() calls inside pytom_match.run() itself, so a
failing combo still can't corrupt another thread's GPU state -- only the
thin Python orchestration around it is now in-process.

Matrix used for the bi38262-30-akinetes screen (2026-08-01), kept here as
a worked example -- edit MATRIX for a new screen's own axes:
  1  baseline            : 70S @ 10.0 A/px, defaults (--angular-search 10)
  2  high-pass 300       : 70S @ 10.0 A/px, --high-pass 300
  3  high-pass 500       : 70S @ 10.0 A/px, --high-pass 500
  4  angular-search 7    : 70S @ 10.0 A/px, --angular-search 7  (finer)
  5  angular-search 15   : 70S @ 10.0 A/px, --angular-search 15 (coarser)
  6  8b0x @ 7.5          : 8b0x @ 7.5 A/px, defaults, low-pass off (Nyquist)
  7  8b0x @ 7.5 + lp20   : 8b0x @ 7.5 A/px, --low-pass 20
  8  tophat off          : 70S @ 10.0 A/px, --tophat-filter omitted

Result of that screen (see the 2026-08-01 conversation for full tables):
baseline+hp300/hp500 and tophat-off were the standout results -- as7 (finer
search) bought more particles but at ~4x the runtime (poor return per
GPU-hour); the 8b0x cells underperformed the 70S baseline outright.
tophat-off measured 2.2x the baseline mean particle count at ~baseline
time cost and passed a visual QC check -- it was subsequently hardcoded
as the new tophat_filter default in pytom_ribo_auto.py's two real
extraction pathways (main run + --reextract).

Angular search is controlled directly via angular_search (cells 4/5),
independent of particle_diameter (fixed at the registry's diameter_a in
every cell, for extraction peak-spacing) -- both are set simultaneously
on the Namespace, which is exactly why this script builds one directly
instead of going through the CLI (see the module-level note above).

Quality metric: particle count from pytom's own auto-estimated LCCmax
cutoff (no --cut-off given at all == "autoselect"), NOT the flat top-N
pytom-ribo-auto defaults to. --number-of-false-positives is fixed at 1
(pytom's own default) in every cell -- it directly controls how lenient
that auto-cutoff is, so letting it vary would confound "better search"
with "looser threshold" and make the comparison meaningless.

Mirror setting: NOT screened here -- fixed to the --check-handedness result
(4-TS majority vote: ts-031, ts-052, ts-082, ts-155 -- 4/4 normal template
wins, no --mirror needed). See MIRROR below.

Volume split: resolution-dependent, NOT a single flat constant (see
VOLUME_SPLIT_BY_APIX) -- [1,1,1] measured ~28% faster than [2,2,1] for the
70S/10.0 A/px entry (fits in GPU memory unsplit: 12107 MiB of 16376 MiB),
but the 8b0x/7.5 A/px entry (~2.4x the voxel count) is projected at ~28 GB
unsplit and would not fit -- kept at the conservative [2,2,1] until that's
actually validated.

Usage:
    python3 pytom_param_screen.py [--dry-run] [--seed 42] [--n-ts 5]
"""

import argparse
import csv
import random
import sys
import threading
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aretomo3_preprocess.commands import pytom_match as _pm
from aretomo3_preprocess.commands.pytom_ribo_auto import (
    _PARTICLES, _pick_bin_variant, _stage_resampled_tomograms,
)

# ── Fixed config -- edit here, not via CLI (mirrors pytom-ribo-auto's own
#    "minimize options" stance for anything that isn't the actual screen axis) ──
PROJECT_DIR   = Path('/mnt/McQueen-001/parry/bi38262-30-akinetes/relion/run001-cmd0-final')
# Curated selection exported from the analyse HTML report's filter sliders
# (overlap 93-98%, defocus 1.39-5.09 um, tilt-axis 85.3-88.8 deg, rating=5
# only) -- 74/172 TS -- rather than the flat rating>=4 ts_select.csv.
TS_SELECT_CSV = Path('/home/sconnell/ts-select_scratchpad_analyse_html_test_'
                     '20260731_1629_ove93-98_def1.39-5.09_til85.3-88.8_rat5-5.csv')
OUT_ROOT      = Path('/mnt/scratch/data-sconnell/pytom_screen')
GPUS          = [0, 1, 2, 3]
EXPECTED_PARTICLES_CAP = 5000   # -n cap, well above what auto-cutoff should ever return
N_FALSE_POSITIVES = 1           # pytom's own default -- fixed, see module docstring

MIRROR = False   # confirmed via --check-handedness, 4/4 TS -- see module docstring

# Resolution-dependent volume-split (see module docstring) -- keyed by the
# registry's target_apix (Angstrom/px), not by particle name.
VOLUME_SPLIT_BY_APIX = {
    10.0: ['1', '1', '1'],
    # 2,1,1 was projected to fit (~14.1 GiB) by a linear fixed+variable
    # memory model fit to the two real 10.0 A/px measurements -- tested
    # for real and it OOMs (CUDA alloc failure at ~16.4 GiB already used,
    # card is 16376 MiB). The model underestimates real GPU-FFT memory
    # scaling (cuFFT workspace/padding costs aren't linear in array size).
    # Back to 2,2,1 (empirically confirmed at 10.0 A/px, not yet directly
    # measured at 7.5 A/px but the safer known-working choice).
    7.5:  ['2', '2', '1'],
}
_DEFAULT_VOLUME_SPLIT = ['2', '2', '1']   # conservative fallback for any other apix

MATRIX = [
    dict(id='01_baseline',    particle='70S',  overrides={}),
    dict(id='02_hp300',       particle='70S',  overrides={'high_pass': 300.0}),
    dict(id='03_hp500',       particle='70S',  overrides={'high_pass': 500.0}),
    dict(id='04_as7',         particle='70S',  overrides={'angular_search': '7'}),
    dict(id='05_as15',        particle='70S',  overrides={'angular_search': '15'}),
    dict(id='06_8b0x_7.5',    particle='8b0x', overrides={}),
    dict(id='07_8b0x_lp20',   particle='8b0x', overrides={'low_pass': 20.0}),
    dict(id='08_notophat',    particle='70S',  overrides={'tophat_filter': False}),
]


def pick_ts(seed, n):
    rows = list(csv.DictReader(open(TS_SELECT_CSV)))
    selected = [r['ts_name'] for r in rows if r['selected'] == '1']
    rng = random.Random(seed)
    chosen = rng.sample(selected, n)
    print(f'Selected {n} TS from {len(selected)} candidates (seed={seed}): {chosen}')
    return chosen


def prepare_tomograms(ts_names, dry_run):
    """Resample the chosen TS to both target apix values ONCE (cached in
    staged dirs), shared across every combo that needs that apix -- avoids
    re-resampling per matrix cell (resample cost dominates otherwise)."""
    staged = {}
    for particle_key in ('70S', '8b0x'):
        reg = _PARTICLES[particle_key]
        target_apix = reg['target_apix']
        if target_apix in staged:
            continue
        vol_suffix, actual_apix, _ = _pick_bin_variant(PROJECT_DIR, target_apix)
        gap_pct = 100.0 * abs(actual_apix - target_apix) / target_apix
        stage_dir = OUT_ROOT / f'staged_tomo_{target_apix:.2f}A'
        if gap_pct > 5.0:
            print(f'Resampling {len(ts_names)} TS to {target_apix:.2f} A/px '
                  f'(closest existing bin {actual_apix:.2f} A/px, {gap_pct:.0f}% off)...')
            new_suffix = _stage_resampled_tomograms(
                PROJECT_DIR, stage_dir, vol_suffix, ts_names, target_apix, dry_run,
            )
            staged[target_apix] = (stage_dir, new_suffix)
        else:
            staged[target_apix] = (PROJECT_DIR, vol_suffix)
    return staged


def build_namespace(cell, ts_name, in_dir, vol_suffix, gpu, out_dir, dry_run=False):
    """Build the argparse.Namespace pytom_match.run() expects, directly --
    NOT via parse_args(), so angular_search and particle_diameter can be
    set together (see module docstring for why the CLI can't do this)."""
    reg = _PARTICLES[cell['particle']]
    apix = reg['target_apix']
    map_key, mask_key = ('map_mirror', 'mask_mirror') if MIRROR else ('map', 'mask')

    o = cell['overrides']
    volume_split = [int(x) for x in VOLUME_SPLIT_BY_APIX.get(apix, _DEFAULT_VOLUME_SPLIT)]
    return argparse.Namespace(
        input=str(in_dir), vol_suffix=vol_suffix,
        select_ts=None, include=[ts_name], exclude=None,
        bmask_dir=None, bmask_suffix='', dose=None,
        template=str(reg[map_key]), mask=str(reg[mask_key]),
        voxel_size=apix, gpu=[gpu],
        particle_diameter=o.get('particle_diameter', reg['diameter_a']),
        angular_search=str(o.get('angular_search', '10')),
        non_spherical_mask=True, z_axis_rotational_symmetry=1,
        volume_split=volume_split, search_x=None, search_y=None, search_z=None,
        tomogram_ctf_model='phase-flip', random_phase_correction=True,
        rng_seed=69, half_precision=True, per_tilt_weighting=True,
        low_pass=o.get('low_pass'), high_pass=o.get('high_pass', 400.0),
        spectral_whitening=False, phase_shift=None, defocus_handedness=None, log=None,
        amplitude_contrast=0.07, spherical_aberration=2.7, voltage=300,
        analyse=True, analyse_thickness=300.0, analyse_output=None,
        extract=True, n_particles=EXPECTED_PARTICLES_CAP,
        tophat_filter=o.get('tophat_filter', True), tophat_bins=None,
        cut_off=None, cut_off_csv=None, n_false_positives=N_FALSE_POSITIVES,
        relion5_compat=True, imod=False, imod_dir=None, imod_sphere_diameter=None,
        output=str(out_dir), extract_only=False, analyse_only=False,
        pytom_dir=None, dry_run=dry_run,
    )


def count_particles(out_dir, ts_name):
    star_files = sorted((out_dir / ts_name).glob('*_particles.star'))
    if not star_files:
        return None
    try:
        import starfile, warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = starfile.read(str(star_files[0]))
        df = data[next(iter(data))] if isinstance(data, dict) else data
        return len(df)
    except Exception as e:
        print(f'  WARNING: could not read {star_files[0]}: {e}')
        return None


def worker(job_q, results, results_lock, gpu, dry_run):
    while True:
        try:
            cell, ts_name, in_dir, vol_suffix = job_q.get_nowait()
        except queue.Empty:
            return
        out_dir = OUT_ROOT / 'runs' / cell['id']
        ns = build_namespace(cell, ts_name, in_dir, vol_suffix, gpu, out_dir, dry_run=dry_run)
        label = f"[gpu{gpu}] {cell['id']} / {ts_name}"
        print(f'{label}: starting'
              f'{"  [angular_search=" + ns.angular_search + "]" if dry_run else ""}')
        try:
            _pm.run(ns)
            n = None if dry_run else count_particles(out_dir, ts_name)
            if not dry_run:
                print(f'{label}: done, {n} particles (auto-cutoff)')
        except SystemExit as e:
            print(f'{label}: FAILED (pytom_match.run() called sys.exit({e.code}))')
            n = None
        except Exception as e:
            print(f'{label}: FAILED ({type(e).__name__}: {e})')
            n = None
        with results_lock:
            results.append({'combo': cell['id'], 'ts': ts_name, 'n_particles': n})
        job_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-ts', type=int, default=5)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ts_names = pick_ts(args.seed, args.n_ts)
    staged = prepare_tomograms(ts_names, args.dry_run)

    job_q = queue.Queue()
    for cell in MATRIX:
        reg = _PARTICLES[cell['particle']]
        in_dir, vol_suffix = staged[reg['target_apix']]
        for ts_name in ts_names:
            job_q.put((cell, ts_name, in_dir, vol_suffix))

    total = job_q.qsize()
    print(f'\n{total} jobs queued across {len(GPUS)} GPUs ({GPUS})\n')

    results = []
    results_lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(job_q, results, results_lock, gpu, args.dry_run))
        for gpu in GPUS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if args.dry_run:
        print('\n[dry-run: no results to summarize]')
        return

    out_csv = OUT_ROOT / 'results.csv'
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['combo', 'ts', 'n_particles'])
        w.writeheader()
        w.writerows(results)
    print(f'\nResults written to {out_csv}')

    print('\nSummary (mean particles per combo):')
    for cell in MATRIX:
        vals = [r['n_particles'] for r in results if r['combo'] == cell['id'] and r['n_particles'] is not None]
        mean = sum(vals) / len(vals) if vals else float('nan')
        print(f"  {cell['id']:16s}  n={len(vals)}/{args.n_ts}  mean={mean:.1f}  values={vals}")


if __name__ == '__main__':
    main()
