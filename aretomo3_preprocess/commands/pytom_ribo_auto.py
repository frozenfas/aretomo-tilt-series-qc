"""
pytom-ribo-auto — fully-automated pytom-match-pick ribosome picking.

A thin, opinionated front-end over `pytom-match`, deliberately kept to a
handful of options: given an AreTomo3 cmd=2 output directory and
--particle, it drives the whole template-matching -> extraction ->
QC-report chain for a bundled ribosome reference, with no template/mask/
voxel-size/particle-diameter bookkeeping left to the user. Only '70S' is
actually usable right now -- see _PARTICLES below.

Every fixed value in this file (the _PARTICLES registry's reference/mask
paths and target_apix, the matching flags in _matching_defaults()) is
hardcoded for THIS lab's (Connell Lab) production conventions and
reference maps -- not a universal default. Adapt _PARTICLES and
_matching_defaults() before reusing this command against a different
lab's references, microscope, or matching conventions.

What it automates that `pytom-match` normally requires by hand
--------------------------------------------------------------
  1. Picks whichever existing `run-aretomo3 --cmd 2` reconstruction (plain
     `_Vol.mrc` or a multi-bin `_bN_Vol.mrc` variant) has a voxel size
     closest to the target 10.0 A/px (fixed, not a CLI option). Resamples
     via IMOD binvol if no existing bin is close enough (see step 1b in
     the code).
  2. Uses the particle registry's pre-made reference + mask directly
     (symlinked into <output>/staged/, no per-run pytom_create_template.py/
     pytom_create_mask.py generation) -- both are required to already
     exist, already be inverted (dark-particle convention) and already be
     at the 10.0 A/px target; see _PARTICLES' docstring. The only
     processing ever run here is --mirror, which can't be done by symlink.
  3. Runs `pytom-match`'s own template-matching + extraction + QC-report
     pipeline (reused directly, not reimplemented) with the flag set
     already validated on this system's real ribosome production runs
     (see `external_pytom2.py` on the BI38262-12-RsmA-cryoET project):
     --per-tilt-weighting --tomogram-ctf-model phase-flip
     --random-phase-correction --half-precision --angular-search 10
     --high-pass 400 --relion5-compat --imod --analyse.

Only 70S has bundled reference/mask pairs on this system right now
---------------------------------------------------------------------
Two candidate 70S references, at two different fixed target_apix, for
comparing against each other (e.g. in a parameter screen):
  '70S'  -- SC_job035_run_it120_class001, a subtomogram average, 10.0 A/px
  '8b0x' -- PDB 8B0X via ChimeraX molmap (native 4.0 A/px), 7.5 A/px

Each has a full mirror pair too (map_mirror/mask_mirror), built with
pytom_create_template.py --mirror (same tool as the originals) from the
same source density, with the mirror mask built independently in RELION
from the resulting mirrored, non-inverted volume -- so mask and template
are co-registered with each other, not just each mirrored separately
(see _prepare_template_and_mask for why that distinction matters).

Add a 'map'/'mask' pair (with a 'target_apix') for 50S/30S/80S to
_PARTICLES to activate those choices; until then they fail with a
clear "no pre-made reference/mask" error rather than silently using the
wrong particle.

Particle diameters (_PARTICLES[...]['diameter_a']) are documented estimates,
not independently verified per-particle -- no CLI override, edit the
registry directly if you have a better number. They drive the extraction
peak-spacing (pytom_extract_candidates.py's own spacing behaviour) and the
matching command's Crowther-criterion angular-search fallback (currently
overridden by the fixed --angular-search 10 in _matching_defaults(), so
this mostly matters for extraction).

Typical usage
-------------
  aretomo3-preprocess pytom-ribo-auto \\
      --input run002-cmd2 --output pytom_ribo_auto --particle 70S

  # Only a subset of TS, flat top-N instead of auto-cutoff
  aretomo3-preprocess pytom-ribo-auto \\
      --input run002-cmd2 --output pytom_ribo_auto --particle 70S \\
      --select-ts run002_analysis/ts-select.csv \\
      --no-auto-cutoff --expected-particles 3000
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
from aretomo3_preprocess.shared.project_state import (
    resolve_selected_ts, record_handedness, resolve_tool_path,
)
from aretomo3_preprocess.shared.discovery import (
    print_cmd as _print_cmd,
    find_volumes as _find_volumes,
    filter_by_include_exclude,
)

_IMOD_DIR   = '/opt/IMOD'
_BINVOL_BIN = f'{_IMOD_DIR}/bin/binvol'

# Target tomogram voxel size -- not exposed as a CLI flag (minimize options
# for this "auto" tool). Per-particle (_PARTICLES[x]['target_apix']), not
# global: different registry entries can be prepared at different fixed
# resolutions (e.g. while screening 10.0 A/px against a finer 7.5 A/px
# reference), each still a single hardcoded value with no from-scratch
# regeneration -- see the 'map'/'mask' comment below for why that's safe.
# Bounded below by whatever the source density's own native resolution
# is -- pytom_create_template.py (>=0.13.2) refuses to upsample past it.
_DEFAULT_TARGET_APIX = 10.0

# How far an existing --at-bin reconstruction is allowed to sit from a
# particle's target_apix before we resample instead of just using it as-is.
_RESAMPLE_TOL_PCT = 5.0

_REF_DIR = Path('/opt/data/pytom')

# diameter_a: envelope diameter in Angstrom, used for Crowther-criterion
# sizing and extraction peak spacing (passed straight through as
# pytom_match_template.py/pytom_extract_candidates.py's own
# --particle-diameter -- no CLI override on our side, see module docstring
# re: these being estimates).
#
# target_apix: fixed voxel size this entry's map/mask were built at --
# NOT a per-tomogram parameter, one value per registry entry. Every run
# for a given particle needs the reference/mask at that exact same voxel
# size, so regenerating them via pytom_create_template.py/
# pytom_create_mask.py on every invocation would recompute the identical
# output every time. Precreating them once and reusing directly (symlink,
# no processing) is strictly equivalent output for less work; there is no
# case where "regenerate every run" would produce a different result,
# since the target never varies for a given entry. This only breaks if an
# entry's target_apix is ever changed after the fact -- then its map/mask
# (and any map_mirror/mask_mirror) need rebuilding at the new value first.
#
# 'map'/'mask' are both REQUIRED to already exist, already inverted
# (dark-particle convention), and already at target_apix -- they are used
# directly as pytom_match_template.py's input, with NO processing
# (rescale/invert/mirror/recenter) ever applied here.
#
# 'map_mirror'/'mask_mirror' (optional, absent unless noted): a SEPARATE,
# independently precreated pair for --mirror/--check-handedness -- same
# reasoning (one target voxel size, so one precreated pair covers every
# future mirror run for this particle), plus there is no runtime mirror
# transform at all (see _prepare_template_and_mask for why: mirroring one
# file without the other risks misaligning them, worse than doing nothing).
#
# Multiple entries can point at the same physical particle (e.g. '70S' vs
# '8b0x' are both full 70S ribosome references, from different sources/at
# different resolutions) -- this is deliberate, not a naming accident: it's
# how a parameter screen compares two candidate references/resolutions
# without either one overwriting the other's registry slot.
_PARTICLES = {
    '70S': {'map': _REF_DIR / 'SC_job035_run_it120_class001_10.00A.mrc',
            'mask': _REF_DIR / 'SC_job035_run_it120_class001_10.00A_MASK.mrc',
            'map_mirror': _REF_DIR / 'SC_job035_run_it120_class001_10.00A_mirror.mrc',
            'mask_mirror': _REF_DIR / 'SC_job035_run_it120_class001_10.00A_mirror_MASK.mrc',
            'diameter_a': 280.0, 'target_apix': 10.0},
    # Same physical particle (70S ribosome) as '70S' above, but from PDB
    # 8B0X via ChimeraX molmap (native 4.0 A/px) rather than the
    # SC_job035 subtomogram average, and at a finer target resolution --
    # a second candidate reference/resolution for the parameter screen,
    # not a replacement for '70S'.
    '8b0x': {'map': _REF_DIR / '8b0x_7.50A_b96.mrc',
             'mask': _REF_DIR / '8b0x_7.50A_b96_noninvert_MASK.mrc',
             'map_mirror': _REF_DIR / '8b0x_7.50A_b96_mirror.mrc',
             'mask_mirror': _REF_DIR / '8b0x_7.50A_b96_mirror_noninvert_MASK.mrc',
             'diameter_a': 290.0, 'target_apix': 7.5},
    '50S': {'map': _REF_DIR / 'map-50S.mrc', 'mask': _REF_DIR / 'mask-50S.mrc',
            'diameter_a': 220.0, 'target_apix': _DEFAULT_TARGET_APIX},
    '30S': {'map': _REF_DIR / 'map-30S.mrc', 'mask': _REF_DIR / 'mask-30S.mrc',
            'diameter_a': 200.0, 'target_apix': _DEFAULT_TARGET_APIX},
    '80S': {'map': _REF_DIR / 'map-80S.mrc', 'mask': _REF_DIR / 'mask-80S.mrc',
            'diameter_a': 300.0, 'target_apix': _DEFAULT_TARGET_APIX},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_voxel_size(mrc_path):
    import mrcfile
    with mrcfile.open(mrc_path, permissive=True) as m:
        return float(m.voxel_size.x)


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


_APIX_MATCH_TOL = 0.01  # A/px; treat as "the same voxel size" within this


def _voxel_size_matches(mrc_path, target_apix):
    try:
        return abs(_read_voxel_size(mrc_path) - target_apix) < _APIX_MATCH_TOL
    except Exception:
        return False


def _use_prescaled_directly(src_path, dst_path, dry_run):
    """Symlink an already-correctly-scaled/processed file straight into
    staged_dir as-is, no pytom_create_template.py/pytom_create_mask.py
    round-trip needed."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f'  [dry-run: would symlink {src_path} -> {dst_path}]')
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.symlink_to(Path(src_path).resolve())


def _check_prescaled(path, label, particle, target_apix):
    """Assumed-precreated reference/mask: must exist and already be at
    target_apix -- errors out rather than silently regenerating, since
    this "auto" tool no longer carries a from-scratch fallback."""
    path = Path(path)
    if not path.exists():
        print(f'ERROR: no pre-made {label} for --particle {particle} (expected {path}).')
        sys.exit(1)
    if not _voxel_size_matches(path, target_apix):
        try:
            have = _read_voxel_size(path)
            have_s = f'{have:.2f} A/px'
        except Exception:
            have_s = 'unreadable'
        print(f'ERROR: {label} {path} is not at the {target_apix:.2f} A/px target '
              f'(has {have_s}). Re-generate it at the target resolution first.')
        sys.exit(1)


def _prepare_template_and_mask(reg, particle, actual_apix, staged_dir, mirror, dry_run):
    """
    Use the pre-created, pre-scaled, pre-inverted reference and mask
    directly (symlinked into staged_dir) as pytom_match_template.py's
    input -- see _PARTICLES' docstring. NO processing (rescale, invert,
    mirror, re-center, ...) is ever applied to either file here; they are
    assumed fully prepared already.

    --mirror requires a SEPARATE pre-made 'map_mirror'/'mask_mirror' pair
    in the registry (built with whatever pipeline made the originals, e.g.
    RELION) rather than a runtime mirror transform -- independently
    mirroring/re-centering just the template (or the mask) via
    pytom_create_template.py risks misaligning the two relative to each
    other (its --center recomputes center of mass from scratch, not
    guaranteed to land at the same position the other file is anchored
    to), which would be worse than doing nothing.

    Shared by the main run and --check-handedness so the two never drift
    apart. Returns (template_path, mask_path), or exits with a clear error
    if the required registry entry is missing/not prepared correctly.
    """
    map_key, mask_key = ('map_mirror', 'mask_mirror') if mirror else ('map', 'mask')
    ref_map  = reg.get(map_key)
    pre_mask = reg.get(mask_key)
    if ref_map is None or pre_mask is None:
        print(f'ERROR: --mirror requested but --particle {particle} has no pre-made '
              f'mirrored reference/mask (_PARTICLES[{particle!r}][{map_key!r}/{mask_key!r}] '
              f'not set). There is no runtime mirror transform -- prepare a properly '
              f'co-registered mirrored pair first (same pipeline used for the originals).')
        sys.exit(1)

    _check_prescaled(ref_map, 'reference', particle, actual_apix)
    _check_prescaled(pre_mask, 'mask', particle, actual_apix)

    staged_dir.mkdir(parents=True, exist_ok=True)
    mirror_tag = '_mirror' if mirror else ''
    template_path = staged_dir / f'template_{particle}_{actual_apix:.2f}A{mirror_tag}.mrc'
    mask_path = staged_dir / f'mask_{particle}_{actual_apix:.2f}A{mirror_tag}.mrc'

    print(f'Using reference {Path(ref_map).name} directly '
          f'(pre-made at {actual_apix:.2f} A/px)')
    _use_prescaled_directly(ref_map, template_path, dry_run)

    print(f'Using mask {Path(pre_mask).name} directly '
          f'(pre-made at {actual_apix:.2f} A/px)')
    _use_prescaled_directly(pre_mask, mask_path, dry_run)

    return template_path, mask_path


def _matching_defaults():
    """
    Fixed pytom_match_template.py flags proven on this system's real
    production ribosome runs -- see external_pytom2.py on the
    BI38262-12-RsmA-cryoET project (a RELION5 External job, not this
    codebase, but the same pytom-match-pick install/convention).

    volume_split=[1,1,1] (no split) measured ~28% faster than [2,2,1] for
    the '70S' registry entry (10.0 A/px, 666x938x293 box): 68m45s vs
    95m44s search time, well within the 16 GB card's memory (12107 MiB
    used unsplit, confirmed stable). volume-split exists purely to keep
    the search volume within GPU memory -- splitting when it isn't needed
    just repeats the full angular search once per sub-volume for no
    benefit. BUT this is resolution-dependent: the '8b0x' entry (7.5 A/px,
    ~2.4x the voxel count) is projected at ~28 GB unsplit -- would NOT fit
    on this card. volume_split isn't yet a per-particle registry field
    (like target_apix is), so if/when 8b0x is actually used for a real
    run, this needs revisiting before assuming [1,1,1] is safe there too.
    """
    return dict(
        angular_search='10', non_spherical_mask=True, z_axis_rotational_symmetry=1,
        volume_split=[1, 1, 1], search_x=None, search_y=None, search_z=None,
        tomogram_ctf_model='phase-flip', random_phase_correction=True,
        rng_seed=69, half_precision=True, per_tilt_weighting=True,
        low_pass=10.0, high_pass=400.0, spectral_whitening=False,
        phase_shift=None, defocus_handedness=None, log=None,
        amplitude_contrast=0.07, spherical_aberration=2.7, voltage=300,
    )


_HANDEDNESS_SEED = 42  # fixed, for a reproducible TS sample across re-runs

# -n cap used under the default auto-cutoff mode when --expected-particles
# isn't given -- well above what auto-cutoff should ever actually return, so
# it never becomes the binding constraint (matches pytom_param_screen.py's
# own EXPECTED_PARTICLES_CAP).
_AUTO_CUTOFF_PARTICLE_CAP = 5000


def _resolve_n_particles(args):
    if not args.auto_cutoff and args.expected_particles is None:
        print('ERROR: --expected-particles is required when --no-auto-cutoff is given')
        sys.exit(1)
    return args.expected_particles if args.expected_particles is not None \
        else _AUTO_CUTOFF_PARTICLE_CAP


def _resolve_handedness_ts_list(in_dir, vol_suffix, select_ts, include, exclude,
                                handedness_ts, n_ts):
    """
    Pick the tomogram(s) to use for --check-handedness -- from the same
    "good ones" (selected_ts) pool pytom_param_screen.py's pick_ts() draws
    from, not just whatever sorts first. TomoGuide recommends 3-4
    tomograms (a single one risks a fluke result), so the default is a
    seeded random sample of n_ts (default 4) from that pool; --handedness-ts
    pins it to one or more specific TS instead (manual override -- these
    are trusted as given and NOT restricted to the selected_ts pool, since
    the whole point is picking exactly what the user wants).
    """
    pairs = _find_volumes(in_dir, vol_suffix)
    all_prefixes = filter_by_include_exclude([p for p, _ in pairs], include, exclude)

    if handedness_ts:
        missing = [ts for ts in handedness_ts if ts not in all_prefixes]
        if missing:
            print(f'ERROR: --handedness-ts {missing} not found in {in_dir} '
                  f'(after --include/--exclude filtering)')
            sys.exit(1)
        print(f'Using {len(handedness_ts)} manually-specified TS: {list(handedness_ts)}')
        return list(handedness_ts)

    prefixes = all_prefixes
    selected_ts = resolve_selected_ts(select_ts)
    if selected_ts is not None:
        prefixes = [p for p in prefixes if p in selected_ts]
    else:
        print('WARNING: --select-ts not given -- sampling from ALL discovered '
              'tomograms, not just the ones marked selected=1 in ts_select.csv. '
              'Pass --select-ts <path to ts_select.csv> to restrict the sample '
              'to vetted TS (recommended).')

    if not prefixes:
        print(f'ERROR: no tomograms found in {in_dir} to check')
        sys.exit(1)

    import random
    n = min(n_ts, len(prefixes))
    chosen = random.Random(_HANDEDNESS_SEED).sample(prefixes, n)
    print(f'Selected {n} TS from {len(prefixes)} candidates '
          f'(seed={_HANDEDNESS_SEED}): {chosen}')
    return chosen


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


def _check_handedness(args, in_dir, out_dir, reg, diameter_a, gpus, sep):
    """
    TomoGuide's handedness workflow (github.com/TomoGuide -- same idea as
    pytom-match-pick's own FAQ, different comparison metric): run matching
    + extraction with both a normal and a mirrored template on several
    tomograms (TomoGuide: "3-4 tomograms that contain ribosomes"; a single
    one risks a fluke result), using the SAME automatic cutoff estimation
    a real run would use, and compare the resulting particle *counts* --
    "one will output way more particles than the other" for the correct
    handedness. Jobs (one per TS x mirror-state) are spread across all
    GPUs in `gpus`, one worker thread per GPU.
    """
    target_apix = reg['target_apix']

    print('Handedness check: normal vs. mirrored template')
    print(sep)

    print('Selecting reconstruction bin...')
    vol_suffix, actual_apix, _sample_vol = _pick_bin_variant(in_dir, target_apix)
    print(sep)

    check_dir = out_dir / 'handedness_check'

    ts_list = _resolve_handedness_ts_list(
        in_dir, vol_suffix, args.select_ts, args.include, args.exclude,
        args.handedness_ts, args.handedness_n_ts,
    )
    print(sep)

    gap_pct = 100.0 * abs(actual_apix - target_apix) / target_apix
    if gap_pct > _RESAMPLE_TOL_PCT:
        vol_suffix = _stage_resampled_tomograms(
            in_dir, check_dir / 'staged_tomo', vol_suffix, ts_list,
            target_apix, args.dry_run, args.imod_bin_dir,
        )
        in_dir      = check_dir / 'staged_tomo'
        actual_apix = target_apix
        print(sep)

    # Template/mask depend only on mirror-state, not on which TS -- prepare
    # each pair once and reuse it for every TS job at that mirror-state.
    staged = {}
    for mirror, label in ((False, 'normal'), (True, 'mirrored')):
        staged[label] = _prepare_template_and_mask(
            reg, args.particle, actual_apix, check_dir / label / 'staged',
            mirror, args.dry_run,
        )
    print(sep)

    def _run_one(ts_name, label, template_path, mask_path, gpu):
        sub_out = check_dir / label / ts_name
        pm_ns = argparse.Namespace(
            input=str(in_dir), vol_suffix=vol_suffix,
            select_ts=None, include=[ts_name], exclude=None,
            bmask_dir=None, bmask_suffix='', dose=None,
            template=str(template_path), mask=str(mask_path),
            voxel_size=actual_apix, gpu=[gpu], particle_diameter=diameter_a,
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
        if args.dry_run:
            return None
        star_files = sorted((sub_out / ts_name).glob('*_particles.star'))
        return _particle_count(star_files[0]) if star_files else None

    jobs = [(ts_name, label) for ts_name in ts_list for label in ('normal', 'mirrored')]
    print(f'Running {len(jobs)} matching jobs ({len(ts_list)} TS x 2 mirror-states) '
          f'across {len(gpus)} GPU(s): {gpus}')
    print(sep)

    results = {ts_name: {} for ts_name in ts_list}
    if len(gpus) <= 1 or args.dry_run:
        # No point threading a single GPU (or a dry-run, which does no real
        # GPU work) -- run sequentially, same result, simpler to read.
        gpu = gpus[0]
        for ts_name, label in jobs:
            template_path, mask_path = staged[label]
            n = _run_one(ts_name, label, template_path, mask_path, gpu)
            results[ts_name][label] = n
            print(sep)
    else:
        import queue
        import threading

        job_q = queue.Queue()
        for job in jobs:
            job_q.put(job)
        results_lock = threading.Lock()

        def _worker(gpu):
            while True:
                try:
                    ts_name, label = job_q.get_nowait()
                except queue.Empty:
                    return
                template_path, mask_path = staged[label]
                print(f'[gpu{gpu}] starting {ts_name} / {label}')
                n = _run_one(ts_name, label, template_path, mask_path, gpu)
                with results_lock:
                    results[ts_name][label] = n
                print(f'[gpu{gpu}] done {ts_name} / {label}: {n} particles')
                job_q.task_done()

        threads = [threading.Thread(target=_worker, args=(gpu,)) for gpu in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(sep)

    if args.dry_run:
        print('[dry-run: skipping particle-count comparison]')
        return

    print(f'Particles extracted (auto cutoff, capped at {args.handedness_particles}):')
    print()
    votes = []
    rows  = []
    for ts_name in ts_list:
        normal_n, mirror_n = results[ts_name].get('normal'), results[ts_name].get('mirrored')
        if normal_n is None or mirror_n is None:
            rows.append((ts_name, '?', '?', 'error (missing star file)'))
            continue
        winner = 'mirrored' if mirror_n > normal_n else 'normal' if normal_n > mirror_n else 'tie'
        votes.append(winner)
        rows.append((ts_name, str(normal_n), str(mirror_n), winner))

    if rows:
        ts_w = max(len('TS'),       *(len(r[0]) for r in rows))
        n_w  = max(len('Normal'),   *(len(r[1]) for r in rows))
        m_w  = max(len('Mirrored'), *(len(r[2]) for r in rows))
        w_w  = max(len('Winner'),   *(len(r[3]) for r in rows))
        rule = f'  {"-"*ts_w}  {"-"*n_w}  {"-"*m_w}  {"-"*w_w}'
        print(f'  {"TS":<{ts_w}}  {"Normal":>{n_w}}  {"Mirrored":>{m_w}}  {"Winner":<{w_w}}')
        print(rule)
        for ts_name, n_s, m_s, winner in rows:
            print(f'  {ts_name:<{ts_w}}  {n_s:>{n_w}}  {m_s:>{m_w}}  {winner:<{w_w}}')
        print(rule)
        print()

    if not votes:
        print('WARNING: no usable results from any TS (starfile not installed? '
              'inspect the star files by hand under '
              f'{check_dir}/{{normal,mirrored}}/<ts>/)')
        return

    n_mirror = votes.count('mirrored')
    n_normal = votes.count('normal')
    n_tie    = votes.count('tie')
    print(sep)
    print(f'Votes: mirrored={n_mirror}  normal={n_normal}  tie={n_tie}  '
          f'(of {len(votes)} usable TS)')

    mirror_recommended = None
    if n_mirror > n_normal:
        mirror_recommended = True
        print('-> majority: mirrored template wins -- re-run the full batch with --mirror')
        print('   Note: this only mirrors the picking TEMPLATE -- the raw '
              'reconstructed tomograms themselves are untouched, and their '
              'physical handedness is opposite to the reference map\'s '
              'true/canonical handedness. You can visually sanity-check this '
              'determination later: the raw tomogram density and a '
              'correctly-handed subtomogram average should look like mirror '
              'images of each other, not the same handedness.')
    elif n_normal > n_mirror:
        mirror_recommended = False
        print('-> majority: normal template wins -- no --mirror needed')
    else:
        print('-> tied vote -- inconclusive, inspect QC/star files by hand before deciding')

    if mirror_recommended is not None:
        record_handedness(
            mirror=mirror_recommended, particle=args.particle,
            per_ts={ts_name: results[ts_name] for ts_name in ts_list},
            source='pytom-ribo-auto --check-handedness',
        )
        print('Recorded in project.json [handedness] for future reference.')


def _reextract(args, out_dir, diameter_a, sep):
    """
    Re-run pytom_extract_candidates.py against an existing --output's
    *_job.json files, without touching matching (no GPU, no --input).
    Lets --expected-particles/--no-auto-cutoff be changed and re-applied
    cheaply -- the GPU search is by far the expensive part.
    """
    if not out_dir.is_dir():
        print(f'ERROR: --output {out_dir} not found (nothing to re-extract from)')
        sys.exit(1)

    n_particles = _resolve_n_particles(args)
    cut_off = None if args.auto_cutoff else 0.0
    print(f'Re-extracting from existing results in {out_dir}')
    print(f'  n_particles={n_particles}  '
          f'cut_off={"auto-estimated" if cut_off is None else cut_off}  '
          f'diameter={diameter_a:.0f} A')
    print(sep)

    pm_ns = argparse.Namespace(
        output=str(out_dir),
        n_particles=n_particles,
        particle_diameter=diameter_a,
        # tophat_filter off: bi38262-30-akinetes param screen (2026-08-01)
        # measured 2.2x more auto-cutoff particles with it off vs baseline,
        # at ~baseline time cost, and a visual QC check of the picks looked
        # fine -- see runs/08_notophat/pytom_extract_qc.html in that screen.
        tophat_filter=False, tophat_bins=None,
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

_SHORT_DESCRIPTION = (
    "Fully-automated pytom-match-pick ribosome picking against a bundled "
    "reference -- runs matching + extraction + QC report end to end from "
    "a run-aretomo3 --cmd 2 output directory. Hardcoded for this lab's "
    "conventions/references (only 70S is bundled right now); see the "
    "module docstring in the code for the full rationale."
)


def add_parser(subparsers):
    p = subparsers.add_parser(
        'pytom-ribo-auto',
        help='Fully-automated pytom-match-pick ribosome picking (bundled reference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=_SHORT_DESCRIPTION,
    )

    req = p.add_argument_group('required')
    req.add_argument('--input', '-i', default=None,
                     help='run-aretomo3 --cmd 2 output directory '
                          '(not required with --reextract)')
    req.add_argument('--output', '-o', required=True,
                     help='Output directory (created; staged/ + per-TS pytom results inside). '
                          'With --reextract, an existing output from a previous run.')
    req.add_argument('--expected-particles', type=int, default=None,
                     help='Flat top-N candidates per tomogram, instead of the default '
                          'auto-cutoff. Required if --no-auto-cutoff is given, ignored '
                          'otherwise.')
    req.add_argument('--no-auto-cutoff', dest='auto_cutoff', action='store_false',
                     help="Extract a flat top-N (needs --expected-particles) instead of "
                          "pytom's automatic ROC-based LCCmax cutoff. Auto-cutoff is the "
                          "default because a flat top-N can't tell a real hit from noise.")

    part = p.add_argument_group('particle')
    part.add_argument('--particle', choices=sorted(_PARTICLES), required=True,
                      help='Which reference to use. Fixes the diameter too -- see '
                           '_PARTICLES in the code for the per-particle values and '
                           'which ones actually have a bundled reference on this system.')
    part.add_argument('--mirror', action='store_true',
                      help='Mirror the reference template before matching '
                           '(chirality/handedness correction -- see '
                           '--check-handedness to determine whether you need this)')

    filt = p.add_argument_group('TS selection')
    filt.add_argument('--select-ts', default=None, metavar='CSV',
                      help='ts-select.csv from select-ts; only selected TS are processed')
    filt.add_argument('--include', nargs='+', help='Process only these TS prefixes')
    filt.add_argument('--exclude', nargs='+', help='Exclude these TS prefixes')

    # --check-handedness follows TomoGuide's handedness workflow ("one will
    # output way more particles than the other"), including its "3-4
    # tomograms" sample-size recommendation (a single TS risks a fluke
    # result) -- majority vote across --handedness-n-ts TS, same
    # auto-estimated cutoff a real run would use. Re-run with --mirror if
    # mirrored wins.
    hand = p.add_argument_group('handedness check')
    hand.add_argument('--check-handedness', action='store_true',
                      help='Compare normal vs. mirrored template on a TS sample '
                           'and exit (does not run the full batch)')
    hand.add_argument('--handedness-ts', nargs='+', default=None, metavar='TS_NAME',
                      help='Check these specific TS instead of a random sample')
    hand.add_argument('--handedness-n-ts', type=int, default=4, metavar='N',
                      help='Sample size if --handedness-ts not given')
    hand.add_argument('--handedness-particles', type=int, default=1000,
                      help='Cap on particles extracted per template for the comparison')

    ctl = p.add_argument_group('run control')
    ctl.add_argument('--reextract', action='store_true',
                     help='Re-run extraction only against an existing --output '
                          '(no --input/GPU needed) -- for changing '
                          '--expected-particles/--no-auto-cutoff cheaply')
    # Default is every visible GPU, which can starve other jobs on shared
    # hardware -- pass this explicitly unless that's really what you want.
    ctl.add_argument('--gpu', '-g', nargs='+', type=int, default=None,
                     help='GPU ID(s) to use (default: auto-detect, ALL visible GPUs)')
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

    args.pytom_dir    = resolve_tool_path('pytom', args.pytom_dir)
    args.imod_bin_dir = resolve_tool_path('imod', args.imod_bin_dir)

    if args.reextract:
        diameter_a = _PARTICLES[args.particle]['diameter_a']
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

    # Fail fast (before disk-space/output-dir/bin-selection/resample) if the
    # registry entries this specific invocation needs aren't there --
    # --mirror/--check-handedness need the separate 'map_mirror'/
    # 'mask_mirror' pair (see _prepare_template_and_mask), not the normal one.
    needed_pairs = [('map', 'mask')]
    if args.mirror or args.check_handedness:
        needed_pairs.append(('map_mirror', 'mask_mirror'))

    def _has_pair(v, map_key, mask_key):
        return v.get(map_key) is not None and Path(v[map_key]).exists() \
           and v.get(mask_key) is not None and Path(v[mask_key]).exists()

    for map_key, mask_key in needed_pairs:
        if not _has_pair(reg, map_key, mask_key):
            available = [k for k, v in _PARTICLES.items()
                         if all(_has_pair(v, mk, sk) for mk, sk in needed_pairs)]
            print(f'ERROR: --particle {args.particle} has no pre-made '
                  f'{map_key}/{mask_key} (needed for '
                  f'{"--check-handedness" if args.check_handedness else "--mirror" if map_key == "map_mirror" else "this run"}).')
            print(f'       Available on this system: {", ".join(available) or "(none)"}')
            print(f'       Add both files to {_REF_DIR}/ to activate this particle.')
            sys.exit(1)

    ref_map = Path(reg['map'])
    target_apix = reg['target_apix']

    diameter_a = reg['diameter_a']

    gpus = args.gpu or _detect_gpus()
    if not gpus:
        print('ERROR: no GPUs given (--gpu) and none auto-detected via nvidia-smi.')
        sys.exit(1)

    print(f'pytom-ribo-auto: {args.particle}  (diameter {diameter_a:.0f} A'
          f'{", mirrored" if args.mirror else ""})')
    if args.gpu:
        print(f'GPUs: {gpus}  (explicit)')
    else:
        print(f'GPUs: {gpus}  (auto-detected -- ALL visible GPUs on this machine; '
              f'pass --gpu to restrict this if others are using it)')
    print(sep)

    if args.check_handedness:
        _check_handedness(args, in_dir, out_dir, reg, diameter_a, gpus, sep)
        return

    for warn_msg in check_disk_space(out_dir):
        print(f'WARNING: {warn_msg}')

    out_dir = check_output_dir(out_dir, clean=args.clean, dry_run=args.dry_run)
    staged_dir = out_dir / 'staged'

    # ── 1. Pick the tomogram bin closest to the target voxel size ──────────
    print('Selecting reconstruction bin...')
    vol_suffix, actual_apix, sample_vol = _pick_bin_variant(in_dir, target_apix)
    print(sep)

    # ── 1b. Resample to the target voxel size if no close-enough bin exists ─
    # No existing --at-bin reconstruction is required to land on the exact
    # target: pytom_match_template.py needs template and tomogram at the
    # *same* voxel size (no internal auto-rescale, unlike easymode), so if
    # the closest bin is more than _RESAMPLE_TOL_PCT off, resample every
    # selected tomogram to the target ourselves rather than searching at a
    # mismatched or merely-close voxel size.
    gap_pct = 100.0 * abs(actual_apix - target_apix) / target_apix
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
            in_dir, staged_dir, vol_suffix, prefixes, target_apix, args.dry_run,
            args.imod_bin_dir,
        )
        in_dir      = staged_dir
        actual_apix = target_apix
        sample_vol  = staged_dir / f'{prefixes[0]}{_RESAMPLED_SUFFIX}_Vol.mrc'
        print(sep)

    # ── 2. Stage reference + mask (pre-made, symlinked -- see _PARTICLES) ───
    template_path, mask_path = _prepare_template_and_mask(
        reg, args.particle, actual_apix, staged_dir, args.mirror, args.dry_run,
    )
    print(sep)

    # ── 3. Stage provenance symlinks ────────────────────────────────────────
    # Keeps the registry reference (identical to template_path unless
    # --mirror, in which case this is the pre-mirror original), the mask,
    # and one representative tomogram at the selected bin all together --
    # open these side by side in ChimeraX to sanity-check contrast/
    # handedness before trusting a full batch run.
    if not args.dry_run:
        _stage_symlink(staged_dir, ref_map)
        if sample_vol.parent != staged_dir:
            _stage_symlink(staged_dir, sample_vol)
        print(f'Staged (provenance): {staged_dir}/')
        print(f'  {ref_map.name}  (registry reference)')
        print(f'  {template_path.name}  (template actually used for matching'
              f'{" -- mirrored" if args.mirror else ", same file as the reference above"})')
        print(f'  {mask_path.name}  (mask)')
        print(f'  {sample_vol.name}  (sample tomogram at this bin)')
        print(sep)

    # ── 4. Delegate to pytom-match's own run() ──────────────────────────────
    # Reuses pytom-match's matching + extraction + QC-report pipeline
    # directly rather than reimplementing it -- see module docstring.
    n_particles = _resolve_n_particles(args)
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
        extract=True, n_particles=n_particles,
        # tophat_filter off: bi38262-30-akinetes param screen (2026-08-01)
        # measured 2.2x more auto-cutoff particles with it off vs baseline,
        # at ~baseline time cost, and a visual QC check of the picks looked
        # fine -- see runs/08_notophat/pytom_extract_qc.html in that screen.
        tophat_filter=False, tophat_bins=None,
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
            'target_apix':  target_apix,
            'actual_apix':  actual_apix,
            'vol_suffix':   vol_suffix,
            'template':     str(template_path),
            'mask':         str(mask_path),
            'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
            'output_dir':   str(out_dir),
        },
        backup_dir=out_dir,
    )
