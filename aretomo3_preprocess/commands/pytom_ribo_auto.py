"""
pytom-ribo-auto — fully-automated pytom-match-pick ribosome picking.

A thin, opinionated front-end over `pytom-match`: given an AreTomo3 cmd=2
output directory and an expected-particle count, it drives the whole
template-matching -> extraction -> QC-report chain for a bundled ribosome
reference (70S/50S/30S/80S), with no template/mask/voxel-size bookkeeping
left to the user.

What it automates that `pytom-match` normally requires by hand
--------------------------------------------------------------
  1. Picks whichever existing `run-aretomo3 --cmd 2` reconstruction (plain
     `_Vol.mrc` or a multi-bin `_bN_Vol.mrc` variant) has a voxel size
     closest to the target 10.0 A/px (fixed, not a CLI option -- matches
     this codebase's other validated ribosome runs, and map-70S.mrc's own
     native 9.06 A/px means finer targets can't be reached anyway --
     pytom_create_template.py refuses to upsample past a reference's
     native resolution). Resamples via IMOD binvol if no existing bin is
     close enough (see step 1b in the code).
  2. Rescales the bundled reference density to that tomogram's *actual*
     measured voxel size (via `pytom_create_template.py`), not to a
     rounded target -- template and tomogram must match pytom-match-pick's
     own voxel size exactly, it does not resample internally the way
     easymode's segmentation models do.
  3. Generates a matching spherical mask (`pytom_create_mask.py`) sized
     from the particle's known diameter.
  4. Symlinks the resolved reference, rescaled template, mask, and the
     tomograms actually used into <output>/staged/ for provenance.
  5. Runs `pytom-match`'s own template-matching + extraction + QC-report
     pipeline (reused directly, not reimplemented) with the flag set
     already validated on this system's real ribosome production runs
     (see `external_pytom2.py` on the BI38262-12-RsmA-cryoET project):
     --per-tilt-weighting --tomogram-ctf-model phase-flip
     --random-phase-correction --half-precision --angular-search 10
     --high-pass 400 --relion5-compat --imod --analyse.

Only 70S has a bundled reference on this system right now
-----------------------------------------------------------
  /opt/data/pytom/map-70S.mrc  (copied from BI38262-12-RsmA-cryoET)
Add map-50S.mrc / map-30S.mrc / map-80S.mrc to the same directory to
activate --particle 50S/30S/80S; until then those choices fail with a
clear "reference not found" error rather than silently using the wrong
particle.

Particle diameters (--particle-diameter) are documented estimates, not
independently verified per-particle -- override them if you have a better
number; they drive both the Crowther-criterion box/mask sizing and the
extraction peak-spacing (pytom_extract_candidates.py's own spacing
behaviour).

Typical usage
-------------
  aretomo3-preprocess pytom-ribo-auto \\
      --input run002-cmd2 \\
      --output pytom_ribo_auto \\
      --expected-particles 3000

  # Only a subset of TS
  aretomo3-preprocess pytom-ribo-auto \\
      --input run002-cmd2 --output pytom_ribo_auto \\
      --expected-particles 3000 --select-ts run002_analysis/ts-select.csv
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
import datetime
from pathlib import Path

from aretomo3_preprocess.commands import pytom_match as _pm
from aretomo3_preprocess.shared.output_guard import check_output_dir, check_disk_space
from aretomo3_preprocess.shared.project_json import update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import resolve_selected_ts
from aretomo3_preprocess.shared.discovery import (
    print_cmd as _print_cmd,
    find_volumes as _find_volumes,
    filter_by_include_exclude,
)

_PYTOM_BIN_DIR          = '/opt/miniconda3/envs/pytom_tm/bin'
_PYTOM_CREATE_TEMPLATE  = f'{_PYTOM_BIN_DIR}/pytom_create_template.py'
_PYTOM_CREATE_MASK      = f'{_PYTOM_BIN_DIR}/pytom_create_mask.py'

_IMOD_DIR   = '/opt/IMOD'
_BINVOL_BIN = f'{_IMOD_DIR}/bin/binvol'

# Target tomogram voxel size -- not exposed as a CLI flag (minimize options
# for this "auto" tool): 10.0 A/px matches this codebase's other validated
# ribosome runs. Coarser than map-70S.mrc's native 9.06 A/px is required --
# pytom_create_template.py (>=0.13.2) refuses to upsample past a
# reference's native resolution.
_TARGET_APIX = 10.0

# How far an existing --at-bin reconstruction is allowed to sit from
# _TARGET_APIX before we resample instead of just using it as-is.
_RESAMPLE_TOL_PCT = 5.0

_REF_DIR = Path('/opt/data/pytom')

# diameter_a: envelope diameter in Angstrom, used for Crowther-criterion
# sizing and extraction peak spacing -- see module docstring re: these
# being estimates. 70S is the only one with a reference file right now.
_PARTICLES = {
    '70S': {'map': _REF_DIR / 'map-70S.mrc', 'diameter_a': 290.0},
    '50S': {'map': _REF_DIR / 'map-50S.mrc', 'diameter_a': 220.0},
    '30S': {'map': _REF_DIR / 'map-30S.mrc', 'diameter_a': 200.0},
    '80S': {'map': _REF_DIR / 'map-80S.mrc', 'diameter_a': 300.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_bin(name, default, pytom_dir=None):
    if pytom_dir:
        c = Path(pytom_dir) / name
        if c.exists():
            return str(c)
    if Path(default).exists():
        return default
    return shutil.which(name) or default


def _read_voxel_size(mrc_path):
    import mrcfile
    with mrcfile.open(mrc_path, permissive=True) as m:
        return float(m.voxel_size.x)


def _read_box_shape(mrc_path):
    import mrcfile
    with mrcfile.open(mrc_path, permissive=True) as m:
        return m.data.shape  # (nz, ny, nx)


def _detect_gpus():
    """Auto-detect GPU indices via nvidia-smi; [] if none found."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True,
        )
        return [int(l.strip()) for l in result.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return []


_VOL_RE = re.compile(r'^(ts-\d+)(_b\d+)?_Vol$')


def _discover_bin_variants(in_dir):
    """
    Return {suffix: sample_vol_path} for every _Vol.mrc bin variant present
    in in_dir.  suffix '' means ts-xxx_Vol.mrc (no bin tag), '_b4' etc for
    multi-bin output.  One sample path per suffix is enough -- all TS from
    the same run-aretomo3 batch share the same --at-bin voxel size.
    """
    variants = {}
    for f in sorted(Path(in_dir).glob('ts-*_Vol.mrc')):
        if '_EVN' in f.name or '_ODD' in f.name:
            continue
        m = _VOL_RE.match(f.stem)
        if not m:
            continue
        suffix = m.group(2) or ''
        variants.setdefault(suffix, f)
    return variants


def _pick_bin_variant(in_dir, target_apix):
    """
    Pick the _Vol.mrc bin variant whose voxel size is closest to
    target_apix.  Returns (vol_suffix, actual_apix, sample_path).
    Exits with an error if no reconstructed volumes exist at all.
    """
    variants = _discover_bin_variants(in_dir)
    if not variants:
        print(f'ERROR: no ts-*_Vol.mrc found in {in_dir} '
              f'(expected a run-aretomo3 --cmd 2 output directory)')
        sys.exit(1)

    scored = []
    for suffix, path in variants.items():
        try:
            apix = _read_voxel_size(path)
        except Exception as e:
            print(f'  WARNING: could not read voxel size from {path.name}: {e}')
            continue
        scored.append((abs(apix - target_apix), suffix, apix, path))

    if not scored:
        print(f'ERROR: found {len(variants)} volume(s) in {in_dir} but could not '
              f'read a voxel size from any of them.')
        sys.exit(1)

    scored.sort(key=lambda t: t[0])
    _, suffix, apix, path = scored[0]

    print(f'  Available bins: '
          + ', '.join(f'{s or "(none)"}={_read_voxel_size(p):.2f} A/px'
                       for s, p in sorted(variants.items()))
          )
    gap_pct = 100.0 * abs(apix - target_apix) / target_apix
    if gap_pct > 20.0:
        print(f'  WARNING: closest available bin is {apix:.2f} A/px, '
              f'{gap_pct:.0f}% away from the requested target {target_apix:.2f} A/px. '
              f'Consider reconstructing an extra bin with '
              f'`run-aretomo3 --cmd 2 --at-bin <N>` closer to the target.')
    print(f'  Using bin: {suffix or "(none)"}  ({apix:.2f} A/px, '
          f'target was {target_apix:.2f} A/px)')
    return suffix, apix, path


def _find_binvol(imod_bin_dir=None):
    if imod_bin_dir:
        c = Path(imod_bin_dir) / 'binvol'
        if c.exists():
            return str(c)
    return shutil.which('binvol') or (_BINVOL_BIN if Path(_BINVOL_BIN).exists() else None)


def _resample_volume(src_path, dst_path, src_apix, target_apix, dry_run, imod_bin_dir=None):
    """Resample one tomogram to target_apix via IMOD binvol (arbitrary float
    binning factor, Lanczos-3 antialiased -- IMOD's own default filter, see
    `binvol -help`) -- unlike run-aretomo3's --at-bin (integer only), this
    hits the target voxel size exactly."""
    binvol_bin = _find_binvol(imod_bin_dir)
    if binvol_bin is None:
        print(f'ERROR: binvol not found (expected {_BINVOL_BIN}). '
              f'Install/locate IMOD (--imod-bin-dir).')
        sys.exit(1)
    factor = target_apix / src_apix
    cmd = [binvol_bin, '-binning', f'{factor:.6f}', '-antialias', '6',
           str(src_path), str(dst_path)]
    if dry_run:
        _print_cmd(cmd)
        print('  [dry-run: skipping execution]')
        return True
    imod_dir = str(Path(binvol_bin).resolve().parent.parent)  # .../bin/binvol -> ...
    env = dict(os.environ, IMOD_DIR=os.environ.get('IMOD_DIR', imod_dir))
    ret = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print(f'\nERROR: binvol failed on {src_path.name}: {ret.stderr.decode().strip()}')
        return False
    return True


_RESAMPLED_SUFFIX = '_resampled'


def _stage_resampled_tomograms(in_dir, staged_dir, vol_suffix, prefixes,
                               target_apix, dry_run, imod_bin_dir=None):
    """
    Resample every selected TS's tomogram to target_apix via IMOD binvol,
    and symlink the aux files (.aln/_CTF.txt/_TLT.txt) pytom-match's
    per-TS metadata reader needs alongside it in staged_dir. Returns the
    vol_suffix to use against staged_dir ('_resampled').

    binvol takes ~1-3 minutes per tomogram (antialiased, arbitrary-factor
    resample of a full volume, not a template-sized box), so this is by
    far the most expensive step for a full batch -- results are cached in
    staged_dir/ across re-runs (skipped if the resampled file already
    exists) so a --reextract or interrupted-and-resumed run doesn't redo it.
    """
    from tqdm import tqdm

    staged_dir.mkdir(parents=True, exist_ok=True)
    print(f'Converting {len(prefixes)} tomogram(s) to the optimal pixel size '
          f'({target_apix:.2f} A/px) via IMOD binvol, Lanczos-3 antialiased '
          f'(~1-3 min each; cached in staged/ across re-runs)...')

    for prefix in tqdm(prefixes, desc='Resampling', unit='tomo'):
        src_vol = _pm._find_tomogram(in_dir, prefix, vol_suffix)
        if src_vol is None:
            tqdm.write(f'  WARNING: no volume for {prefix} -- skipping')
            continue
        dst_vol = staged_dir / f'{prefix}{_RESAMPLED_SUFFIX}_Vol.mrc'
        if dst_vol.exists() and not dry_run:
            tqdm.write(f'  {prefix}: already at {target_apix:.2f} A/px, skipping '
                       f'(delete to force regeneration)')
        else:
            src_apix = _read_voxel_size(src_vol)
            tqdm.write(f'  {prefix}: {src_apix:.2f} -> {target_apix:.2f} A/px (optimal)')
            if not _resample_volume(src_vol, dst_vol, src_apix, target_apix,
                                    dry_run, imod_bin_dir):
                sys.exit(1)

        for aux_suffix in ('.aln', '_CTF.txt', '_TLT.txt'):
            src_aux = in_dir / f'{prefix}{aux_suffix}'
            if src_aux.exists():
                _stage_symlink(staged_dir, src_aux)

    return _RESAMPLED_SUFFIX


def _stage_symlink(dst_dir, src_path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(src_path).name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path(src_path).resolve())
    return dst


def _run_or_print(cmd, dry_run, log_label):
    _print_cmd(cmd)
    if dry_run:
        print('  [dry-run: skipping execution]')
        return True
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f'  ERROR: {log_label} exited with code {ret.returncode}')
        return False
    return True


def _prepare_template_and_mask(ref_map, particle, diameter_a, actual_apix,
                               staged_dir, sigma, mirror, pytom_dir, dry_run):
    """
    Rescale ref_map to actual_apix (inverted, optionally mirrored) and
    generate a matching spherical mask.  Shared by the main run and
    --check-handedness so the two never drift apart.

    Returns (template_path, mask_path), or exits on a subprocess failure.
    """
    create_template_bin = _find_bin('pytom_create_template.py', _PYTOM_CREATE_TEMPLATE, pytom_dir)
    create_mask_bin      = _find_bin('pytom_create_mask.py',     _PYTOM_CREATE_MASK,     pytom_dir)

    # --invert: the raw reference is a standard positive-density map (bright
    # protein in ChimeraX' default convention), but this codebase's
    # AreTomo3/SART reconstructions have DARK particles on a bright
    # background (verified empirically against real picked ribosome
    # coordinates in BI38262-12-RsmA-cryoET: particle sites average ~0.37
    # std below the tomogram's global mean) -- so the template must be
    # sign-flipped to match. --mirror is the separate chirality/handedness
    # axis (see --check-handedness).
    mirror_tag = '_mirror' if mirror else ''
    print(f'Rescaling reference {ref_map.name} -> {actual_apix:.2f} A/px'
          f'{" (mirrored)" if mirror else ""}...')
    template_path = staged_dir / f'template_{particle}_{actual_apix:.2f}A{mirror_tag}.mrc'
    tmpl_cmd = [
        create_template_bin,
        '-i', str(ref_map),
        '--output-voxel-size-angstrom', str(actual_apix),
        '--invert',
        '--center',
        '-o', str(template_path),
    ]
    if mirror:
        tmpl_cmd.append('--mirror')
    staged_dir.mkdir(parents=True, exist_ok=True)
    if not _run_or_print(tmpl_cmd, dry_run, 'pytom_create_template.py'):
        sys.exit(1)

    if dry_run and not template_path.exists():
        # Can't read the not-yet-generated template's box size; fall back to
        # the raw reference's own box (rescale changes it, but this is only
        # for printing a representative dry-run command).
        box_size = _read_box_shape(ref_map)[0]
    else:
        box_size = _read_box_shape(template_path)[0]
    radius_px = max(1, round((diameter_a / 2.0) / actual_apix))

    print(f'Generating mask (box {box_size}px, radius {radius_px}px, sigma {sigma})...')
    mask_path = staged_dir / f'mask_{particle}_{actual_apix:.2f}A.mrc'
    mask_cmd = [
        create_mask_bin,
        '-b', str(box_size),
        '-r', str(radius_px),
        '-s', str(sigma),
        '--voxel-size', str(actual_apix),
        '-o', str(mask_path),
    ]
    if not _run_or_print(mask_cmd, dry_run, 'pytom_create_mask.py'):
        sys.exit(1)

    return template_path, mask_path


def _matching_defaults():
    """
    Fixed pytom_match_template.py flags proven on this system's real
    production ribosome runs -- see external_pytom2.py on the
    BI38262-12-RsmA-cryoET project (a RELION5 External job, not this
    codebase, but the same pytom-match-pick install/convention).
    """
    return dict(
        angular_search='10', non_spherical_mask=True, z_axis_rotational_symmetry=1,
        volume_split=[2, 2, 1], search_x=None, search_y=None, search_z=None,
        tomogram_ctf_model='phase-flip', random_phase_correction=True,
        rng_seed=69, half_precision=True, per_tilt_weighting=True,
        low_pass=10.0, high_pass=400.0, spectral_whitening=False,
        phase_shift=None, defocus_handedness=None, log=None,
        amplitude_contrast=0.07, spherical_aberration=2.7, voltage=300,
    )


def _resolve_default_ts(in_dir, vol_suffix, select_ts, include, exclude, handedness_ts):
    """Pick the tomogram to use for --check-handedness."""
    pairs = _find_volumes(in_dir, vol_suffix)
    prefixes = [p for p, _ in pairs]
    prefixes = filter_by_include_exclude(prefixes, include, exclude)
    selected_ts = resolve_selected_ts(select_ts)
    if selected_ts is not None:
        prefixes = [p for p in prefixes if p in selected_ts]

    if handedness_ts:
        if handedness_ts not in prefixes:
            print(f'ERROR: --handedness-ts {handedness_ts} not found in {in_dir} '
                  f'(after --select-ts/--include/--exclude filtering)')
            sys.exit(1)
        return handedness_ts

    if not prefixes:
        print(f'ERROR: no tomograms found in {in_dir} to check')
        sys.exit(1)
    return prefixes[0]


def _particle_count(star_path):
    """Row count of a pytom-extracted particles STAR file, or None."""
    try:
        import starfile
        import warnings
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        data = starfile.read(str(star_path))
    df = data[next(iter(data))] if isinstance(data, dict) else data
    return len(df)


def _check_handedness(args, in_dir, out_dir, ref_map, diameter_a, gpus, sep):
    """
    TomoGuide's handedness workflow (github.com/TomoGuide -- same idea as
    pytom-match-pick's own FAQ, different comparison metric): run matching
    + extraction with both a normal and a mirrored template on one
    tomogram, using the SAME automatic cutoff estimation a real run would
    use, and compare the resulting particle *counts* -- "one will output
    way more particles than the other" for the correct handedness.
    """
    print('Handedness check: normal vs. mirrored template on one tomogram')
    print(sep)

    print('Selecting reconstruction bin...')
    vol_suffix, actual_apix, _sample_vol = _pick_bin_variant(in_dir, _TARGET_APIX)
    print(sep)

    check_dir = out_dir / 'handedness_check'

    ts_name = _resolve_default_ts(
        in_dir, vol_suffix, args.select_ts, args.include, args.exclude,
        args.handedness_ts,
    )
    print(f'Using tomogram: {ts_name}')
    print(sep)

    gap_pct = 100.0 * abs(actual_apix - _TARGET_APIX) / _TARGET_APIX
    if gap_pct > _RESAMPLE_TOL_PCT:
        vol_suffix = _stage_resampled_tomograms(
            in_dir, check_dir / 'staged_tomo', vol_suffix, [ts_name],
            _TARGET_APIX, args.dry_run, args.imod_bin_dir,
        )
        in_dir      = check_dir / 'staged_tomo'
        actual_apix = _TARGET_APIX
        print(sep)

    results = {}

    for mirror, label in ((False, 'normal'), (True, 'mirrored')):
        print(f'-- {label} template --')
        sub_out = check_dir / label
        template_path, mask_path = _prepare_template_and_mask(
            ref_map, args.particle, diameter_a, actual_apix, sub_out / 'staged',
            args.sigma, mirror, args.pytom_dir, args.dry_run,
        )

        pm_ns = argparse.Namespace(
            input=str(in_dir), vol_suffix=vol_suffix,
            select_ts=None, include=[ts_name], exclude=None,
            bmask_dir=None, bmask_suffix='', dose=None,
            template=str(template_path), mask=str(mask_path),
            voxel_size=actual_apix, gpu=gpus, particle_diameter=diameter_a,
            **_matching_defaults(),
            analyse=False, analyse_thickness=300.0, analyse_output=None,
            extract=True, n_particles=args.handedness_particles,
            tophat_filter=True, tophat_bins=None,
            cut_off=None, cut_off_csv=None, n_false_positives=None,
            relion5_compat=True, imod=False, imod_dir=None, imod_sphere_diameter=None,
            output=str(sub_out), extract_only=False, analyse_only=False,
            pytom_dir=args.pytom_dir, dry_run=args.dry_run,
        )
        _pm.run(pm_ns)

        if not args.dry_run:
            star_files = sorted((sub_out / ts_name).glob('*_particles.star'))
            if star_files:
                results[label] = _particle_count(star_files[0])
        print(sep)

    if args.dry_run:
        print('[dry-run: skipping particle-count comparison]')
        return

    normal_n, mirror_n = results.get('normal'), results.get('mirrored')
    if normal_n is not None and mirror_n is not None:
        print(f'Particles extracted (auto cutoff, capped at {args.handedness_particles}): '
              f'normal={normal_n}  mirrored={mirror_n}')
        if mirror_n > normal_n:
            print('-> mirrored template extracted more particles: '
                  're-run the full batch with --mirror')
        elif normal_n > mirror_n:
            print('-> normal template extracted more particles: no --mirror needed')
        else:
            print('-> tied -- inconclusive, inspect both QC/star files by hand')
    else:
        print('WARNING: could not read particle counts from one or both results '
              '(starfile not installed?) -- inspect the star files by hand:')
        print(f'  {check_dir}/normal/{ts_name}/*_particles.star')
        print(f'  {check_dir}/mirrored/{ts_name}/*_particles.star')


def _reextract(args, out_dir, diameter_a, sep):
    """
    Re-run pytom_extract_candidates.py against an existing --output's
    *_job.json files, without touching matching (no GPU, no --input).
    Lets --expected-particles/--auto-cutoff/--particle-diameter be changed
    and re-applied cheaply -- the GPU search is by far the expensive part.
    """
    if not out_dir.is_dir():
        print(f'ERROR: --output {out_dir} not found (nothing to re-extract from)')
        sys.exit(1)

    cut_off = None if args.auto_cutoff else 0.0
    print(f'Re-extracting from existing results in {out_dir}')
    print(f'  n_particles={args.expected_particles}  '
          f'cut_off={"auto-estimated" if cut_off is None else cut_off}  '
          f'diameter={diameter_a:.0f} A')
    print(sep)

    pm_ns = argparse.Namespace(
        output=str(out_dir),
        n_particles=args.expected_particles,
        particle_diameter=diameter_a,
        tophat_filter=True, tophat_bins=None,
        cut_off=cut_off, cut_off_csv=None, n_false_positives=None,
        relion5_compat=True, imod=True, imod_dir=None, imod_sphere_diameter=None,
        select_ts=args.select_ts, include=args.include, exclude=args.exclude,
        analyse=True, analyse_thickness=300.0, analyse_output=None,
        pytom_dir=args.pytom_dir, dry_run=args.dry_run, log=None,
        extract_only=True, analyse_only=False,
    )
    _pm.run(pm_ns)


# ─────────────────────────────────────────────────────────────────────────────
# Parser registration
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'pytom-ribo-auto',
        help='Fully-automated pytom-match-pick ribosome picking (bundled reference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    req = p.add_argument_group('required')
    req.add_argument('--input', '-i', default=None,
                     help='run-aretomo3 --cmd 2 output directory '
                          '(not required with --reextract)')
    req.add_argument('--output', '-o', required=True,
                     help='Output directory (created; staged/ + per-TS pytom results inside). '
                          'With --reextract, an existing output from a previous run.')
    req.add_argument('--expected-particles', type=int, required=True,
                     help='Number of top-scoring candidates to extract per tomogram. '
                          'Cut-off is disabled (0) by default, so this picks the top-N '
                          'candidates directly rather than filtering by an '
                          'auto-estimated LCCmax significance threshold -- see '
                          '--auto-cutoff to restore that behaviour instead.')
    req.add_argument('--auto-cutoff', action='store_true',
                     help='Use pytom_extract_candidates.py\'s automatic ROC-based '
                          'LCCmax cutoff estimation instead of extracting a flat '
                          'top-N. Can return 0 particles even for real hits if the '
                          'estimated cutoff is conservative -- off by default for '
                          'that reason.')

    part = p.add_argument_group('particle')
    part.add_argument('--particle', choices=sorted(_PARTICLES), default='70S',
                      help='Which reference to use (only 70S has a bundled '
                           'reference on this system right now)')
    part.add_argument('--particle-diameter', type=float, default=None,
                      help='Override the built-in diameter estimate (Angstrom)')
    part.add_argument('--mirror', action='store_true',
                      help='Mirror the reference template before matching '
                           '(chirality/handedness correction -- see '
                           '--check-handedness to determine whether you need this)')

    filt = p.add_argument_group('TS selection')
    filt.add_argument('--select-ts', default=None, metavar='CSV',
                      help='ts-select.csv from select-ts; only selected TS are processed')
    filt.add_argument('--include', nargs='+', help='Process only these TS prefixes')
    filt.add_argument('--exclude', nargs='+', help='Exclude these TS prefixes')

    tgt = p.add_argument_group('target resolution')
    tgt.add_argument('--sigma', type=float, default=1.0,
                     help='Mask edge fall-off (px); 0.5-1.0 recommended for 10-20 A/px')

    hand = p.add_argument_group('handedness check')
    hand.add_argument('--check-handedness', action='store_true',
                      help='Run matching+extraction with BOTH the normal and '
                           'mirrored template on a single tomogram, report which '
                           'extracts more particles under the same auto-estimated '
                           'cutoff a real run would use, and exit -- does not run '
                           'the full batch. Follows TomoGuide\'s handedness workflow '
                           '("one will output way more particles than the other"). '
                           'Re-run with --mirror if the mirrored version wins.')
    hand.add_argument('--handedness-ts', default=None, metavar='TS_NAME',
                      help='Which tomogram to use for --check-handedness '
                           '(default: first one found/selected)')
    hand.add_argument('--handedness-particles', type=int, default=1000,
                      help='Cap on particles extracted per template for the '
                           'comparison (auto-estimated cutoff still applies, same '
                           'as a real run -- this is just an upper bound)')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--reextract', action='store_true',
                     help='Skip matching entirely and re-run extraction only, '
                          'against the existing --output from a previous run -- '
                          'use this to change --expected-particles/--auto-cutoff/'
                          '--particle-diameter without re-running the GPU search. '
                          'Does not need --input or a GPU.')
    ctl.add_argument('--gpu', '-g', nargs='+', type=int, default=None,
                     help='GPU ID(s) to use. REQUIRED to think about on shared '
                          'hardware: default is to auto-detect and use *all* '
                          'visible GPUs, which can starve other jobs on this '
                          'machine -- pass this explicitly (e.g. --gpu 0 1) '
                          'unless you really want every GPU.')
    ctl.add_argument('--clean', action='store_true',
                     help='Remove an existing --output directory before starting')
    ctl.add_argument('--pytom-dir', default=None,
                     help='Directory containing pytom binaries '
                          '(default: /opt/miniconda3/envs/pytom_tm/bin/)')
    ctl.add_argument('--imod-bin-dir', default=None,
                     help='Directory containing IMOD binaries (binvol), used when '
                          'resampling to the 10.0 A/px target '
                          '(default: /opt/IMOD/bin/, or binvol on PATH)')
    ctl.add_argument('--dry-run', action='store_true',
                     help='Print every command without running anything')

    p.set_defaults(func=run)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    sep = '─' * 70
    out_dir = Path(args.output).resolve()

    if args.reextract:
        diameter_a = args.particle_diameter or _PARTICLES[args.particle]['diameter_a']
        _reextract(args, out_dir, diameter_a, sep)
        return

    if args.input is None:
        print('ERROR: --input is required unless --reextract is given')
        sys.exit(1)
    in_dir = Path(args.input).resolve()

    if not in_dir.is_dir():
        print(f'ERROR: --input {in_dir} not found')
        sys.exit(1)

    reg = _PARTICLES[args.particle]
    ref_map = Path(reg['map'])
    if not ref_map.exists():
        available = [k for k, v in _PARTICLES.items() if Path(v['map']).exists()]
        print(f'ERROR: no reference map for --particle {args.particle} '
              f'(expected {ref_map}).')
        print(f'       Available on this system: {", ".join(available) or "(none)"}')
        print(f'       Add the map to {_REF_DIR}/ to activate this particle.')
        sys.exit(1)

    diameter_a = args.particle_diameter or reg['diameter_a']

    gpus = args.gpu or _detect_gpus()
    if not gpus:
        print('ERROR: no GPUs given (--gpu) and none auto-detected via nvidia-smi.')
        sys.exit(1)

    print(f'pytom-ribo-auto: {args.particle}  (diameter {diameter_a:.0f} A'
          f'{" -- override" if args.particle_diameter else " -- built-in estimate"}'
          f'{", mirrored" if args.mirror else ""})')
    if args.gpu:
        print(f'GPUs: {gpus}  (explicit)')
    else:
        print(f'GPUs: {gpus}  (auto-detected -- ALL visible GPUs on this machine; '
              f'pass --gpu to restrict this if others are using it)')
    print(sep)

    if args.check_handedness:
        _check_handedness(args, in_dir, out_dir, ref_map, diameter_a, gpus, sep)
        return

    for warn_msg in check_disk_space(out_dir):
        print(f'WARNING: {warn_msg}')

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)
    staged_dir = out_dir / 'staged'

    # ── 1. Pick the tomogram bin closest to the target voxel size ──────────
    print('Selecting reconstruction bin...')
    vol_suffix, actual_apix, sample_vol = _pick_bin_variant(in_dir, _TARGET_APIX)
    print(sep)

    # ── 1b. Resample to the target voxel size if no close-enough bin exists ─
    # No existing --at-bin reconstruction is required to land on the exact
    # target: pytom_match_template.py needs template and tomogram at the
    # *same* voxel size (no internal auto-rescale, unlike easymode), so if
    # the closest bin is more than _RESAMPLE_TOL_PCT off, resample every
    # selected tomogram to the target ourselves rather than searching at a
    # mismatched or merely-close voxel size.
    gap_pct = 100.0 * abs(actual_apix - _TARGET_APIX) / _TARGET_APIX
    if gap_pct > _RESAMPLE_TOL_PCT:
        prefixes = [p for p, _ in _find_volumes(in_dir, vol_suffix)]
        prefixes = filter_by_include_exclude(prefixes, args.include, args.exclude)
        selected_ts = resolve_selected_ts(args.select_ts)
        if selected_ts is not None:
            prefixes = [p for p in prefixes if p in selected_ts]
        if not prefixes:
            print('ERROR: no tomograms left to resample after filtering')
            sys.exit(1)

        vol_suffix = _stage_resampled_tomograms(
            in_dir, staged_dir, vol_suffix, prefixes, _TARGET_APIX, args.dry_run,
            args.imod_bin_dir,
        )
        in_dir      = staged_dir
        actual_apix = _TARGET_APIX
        sample_vol  = staged_dir / f'{prefixes[0]}{_RESAMPLED_SUFFIX}_Vol.mrc'
        print(sep)

    # ── 2-4. Rescale reference + generate matching mask ─────────────────────
    template_path, mask_path = _prepare_template_and_mask(
        ref_map, args.particle, diameter_a, actual_apix, staged_dir,
        args.sigma, args.mirror, args.pytom_dir, args.dry_run,
    )
    print(sep)

    # ── 5. Stage provenance symlinks ────────────────────────────────────────
    # Keeps the raw reference, the rescaled/inverted template actually
    # searched with, the mask, and one representative tomogram at the
    # selected bin all together -- open these side by side in ChimeraX to
    # sanity-check contrast/handedness before trusting a full batch run.
    if not args.dry_run:
        _stage_symlink(staged_dir, ref_map)
        if sample_vol.parent != staged_dir:
            _stage_symlink(staged_dir, sample_vol)
        print(f'Staged (provenance): {staged_dir}/')
        print(f'  {ref_map.name}  (raw reference, as-is)')
        print(f'  {template_path.name}  (rescaled + inverted template actually used)')
        print(f'  {mask_path.name}  (mask)')
        print(f'  {sample_vol.name}  (sample tomogram at this bin)')
        print(sep)

    # ── 6. Delegate to pytom-match's own run() ──────────────────────────────
    # Reuses pytom-match's matching + extraction + QC-report pipeline
    # directly rather than reimplementing it -- see module docstring.
    pm_ns = argparse.Namespace(
        # input
        input=str(in_dir), vol_suffix=vol_suffix,
        select_ts=args.select_ts, include=args.include, exclude=args.exclude,
        bmask_dir=None, bmask_suffix='', dose=None,
        # template matching
        template=str(template_path), mask=str(mask_path),
        voxel_size=actual_apix, gpu=gpus, particle_diameter=diameter_a,
        **_matching_defaults(),
        # QC report
        analyse=True, analyse_thickness=300.0, analyse_output=None,
        # extraction
        extract=True, n_particles=args.expected_particles,
        tophat_filter=True, tophat_bins=None,
        cut_off=(None if args.auto_cutoff else 0.0),
        cut_off_csv=None, n_false_positives=None,
        relion5_compat=True, imod=True, imod_dir=None, imod_sphere_diameter=None,
        # run control
        output=str(out_dir), extract_only=False, analyse_only=False,
        pytom_dir=args.pytom_dir, dry_run=args.dry_run,
    )

    print('Running pytom-match (matching + extraction + QC report)...')
    print(sep)
    _pm.run(pm_ns)

    if args.dry_run:
        return

    print(sep)
    match_qc  = out_dir / 'pytom_match_qc.html'
    extract_qc = out_dir / 'pytom_extract_qc.html'
    print('Done.')
    print(f'  Particle star files : {out_dir}/*/*_particles.star')
    if match_qc.exists():
        print(f'  Match QC (browser)   : {match_qc}')
    if extract_qc.exists():
        print(f'  Picks QC (browser)   : {extract_qc}')

    update_section(
        section='pytom_ribo_auto',
        values={
            'command':      ' '.join(sys.argv),
            'args':         args_to_dict(args),
            'particle':     args.particle,
            'diameter_a':   diameter_a,
            'target_apix':  _TARGET_APIX,
            'actual_apix':  actual_apix,
            'vol_suffix':   vol_suffix,
            'template':     str(template_path),
            'mask':         str(mask_path),
            'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
            'output_dir':   str(out_dir),
        },
        backup_dir=out_dir,
    )
