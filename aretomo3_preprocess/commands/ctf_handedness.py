"""
ctf-handedness — automated defocus-handedness QC via IMOD ctfplotter -testInv.

Determines the rlnTomoHand ±1 convention relion5-convert needs: IMOD's own
ctfplotter has a purpose-built test for this, `-testInv`, documented in
Xiong et al. 2009 J. Struct. Biol. and the ctfplotter man page. It measures
the defocus difference between the left/right sides of the (raw, unaligned)
tilt series at a high-tilt view set and reports whether the tilt angles'
sign is consistent with that gradient:

    Angle polarity needed: x  ratio: yy.yy

x=1 means the tilt angles as given are correct (no inversion needed), x=-1
means they need inverting, x=0 is ambiguous. ratio is bigger/smaller
left-right defocus difference; the program's own criterion for a "clear"
result is ratio >= 2.5.

Mapping ctfplotter's x to the RELION ±1 convention (derived from IMOD's own
docs, not assumed): the IMOD manual says "At positive tilt angles, the
right side should be more underfocused than the left side" when x=1 (no
inversion needed) -- i.e. right increasing / left decreasing defocus with
tilt. Cross-referenced against the DefocusGrad tool's (CellArchLab/
cryoet-scripts) own published convention -- "LEFT side DECREASING defocus
towards positive tilt, RIGHT side INCREASING -> RELION -1; opposite -> +1"
-- the same physical pattern, giving:
    ctfplotter x=+1  <->  RELION -1
    ctfplotter x=-1  <->  RELION +1
    ctfplotter x=0   <->  inconclusive
_polarity_to_relion() below encodes this so the report always shows the
RELION-convention number, not raw ctfplotter output.

ctfplotter -testInv itself doesn't persist a plottable per-tilt defocus
curve (it prints a result and exits). To get a DefocusGrad-style plot
(defocus vs tilt, left=blue/right=orange) for visual inspection, this
command optionally (on by default, --skip-plots to disable) builds
left/right aligned half-stacks the same way DefocusGrad's own script does
(newstack -xform/-offset/-bin, tilt axis vertical after alignment) and runs
`ctfplotter -autoFit 0,0` (per-view fit) on each half separately, using a
-scan range centered on this TS's own already-known mean_defocus_um from
alignment_data.json rather than a blind wide default -- verified this
matters: a blind 5-60um scan (DefocusGrad's own CTFFIND default range)
produced garbage single-view ctfplotter fits (fit error ~0.2, defocus
hundreds of um), while a scan centered on the known ~1.9um true defocus
gave clean fits (error ~0.006) tracking the known range exactly. The plot
also shows a second panel: delta = right - left defocus at matching tilts,
with its own linear fit (_make_plot). This matters because the raw
per-side defocus vs tilt is often not flat even ignoring handedness --
confirmed on real data (ts-083) that AreTomo3's own overall per-frame
defocus estimate ranges ~15000-34000 A across one tilt series (a common-
mode trend from something other than the left-right gradient: stage/
autofocus drift, poor eucentricity, etc.), which swamped the smaller
differential signal enough that the two raw per-side slopes came out
same-sign ("inconclusive" by the naive slope-sign test) even though
-testInv gave a clear, correct result and the plot visually showed the
right pattern. Subtracting right-left cancels any such common-mode trend
at each matching tilt, isolating the differential (handedness) signal --
delta_slope/delta_corr/delta_relion_handedness are exposed per-TS, still
only as a secondary/diagnostic cross-check (never the primary verdict,
which is always -testInv's own result). shared/tilt_series_qc.py's TS
auto-selection now also screens for this: it scores each candidate TS's
AreTomo3-reported defocus_tilt_spread_um (std of mean_defocus_um across
that TS) and ranks flatter TS higher, since a flat TS is inherently less
likely to need this correction in the first place.

Confirmed via a real subprocess test that the installed /opt/IMOD/bin/
ctfplotter (a full Qt build, not the "standalone no-Qt" build the manual
describes) runs correctly non-interactively with QT_QPA_PLATFORM=offscreen
and no $DISPLAY -- exits 0, no hang, no GUI required. This was a direct,
unverified risk before implementing (an exact precedent to DefocusGrad's
own plt.show()-hang bug encountered while evaluating that tool) --
confirmed safe before relying on it.

Example
-------
  aretomo3-preprocess ctf-handedness \\
      --analysis run002-cmd1/analyse --n-ts 4 --output ctf_handedness_qc
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import mrcfile

from aretomo3_preprocess.shared.parsers import parse_aln_file
from aretomo3_preprocess.shared.project_json import load as load_project, update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import (
    get_latest_analysis_dir, get_cmd0_outdir, get_run_params,
    resolve_tool_path, record_analysis_run,
)
from aretomo3_preprocess.shared.landing_page import write_landing_page
from aretomo3_preprocess.shared.output_guard import check_output_dir
from aretomo3_preprocess.shared.tilt_series_qc import select_ts as _select_ts, imod_env as _imod_env

_DEFAULT_IMOD_DIR = '/opt/IMOD'

# ctfplotter's own "clear result" criterion (see man page: -testInv).
_CLEAR_RATIO = 2.5

# ctfplotter x -> RELION/DefocusGrad tilt-handedness convention -- see
# module docstring for the derivation from both tools' own documentation.
_POLARITY_TO_RELION = {1: -1, -1: 1, 0: 0}

_RE_POLARITY   = re.compile(r'Angle polarity needed:\s*(-?\d+)\s*ratio:\s*([\d.]+)')
_RE_LR_ASGIVEN = re.compile(r'With angles in tilt file\s*:\s*left\s*([\d.]+)\s*right\s*([\d.]+)\s*difference\s*(-?[\d.]+)')
_RE_LR_INVERT  = re.compile(r'With angles inverted:\s*left\s*([\d.]+)\s*right\s*([\d.]+)\s*difference\s*(-?[\d.]+)')


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apix_A(mrc_path) -> float:
    with mrcfile.open(str(mrc_path), permissive=True, header_only=True) as mrc:
        return float(mrc.voxel_size.x)


def _scan_range_nm(ts_data: dict, pad_um: float):
    """(-scan lo,hi) in nm centred on this TS's own already-known
    mean_defocus_um (alignment_data.json's per-frame CTFFIND result) rather
    than a blind wide default -- confirmed via a real test that ctfplotter's
    per-view -autoFit 0,0 fit is unreliable with a wide blind scan range but
    clean when centred on the known true defocus (see module docstring)."""
    defoci = [f['mean_defocus_um'] for f in ts_data.get('frames', [])
              if f.get('mean_defocus_um') is not None]
    center_um = float(np.median(defoci)) if defoci else 2.0
    lo = max(200.0, (center_um - pad_um) * 1000.0)
    hi = (center_um + pad_um) * 1000.0
    return lo, hi, center_um


def _relion_verdict_line(parsed: dict) -> str:
    """Human-readable "Import into RELION using ... handedness" sentence,
    matching DefocusGrad's own CONCLUSION-line phrasing (that's the
    established convention users of this pipeline already recognise) --
    derived from ctfplotter's own x/ratio via _POLARITY_TO_RELION, NOT
    printed by ctfplotter itself. Always spells out "(RELION convention,
    converted from ctfplotter x=...)" explicitly -- ctfplotter's raw x
    uses the opposite sign (x=+1 <-> RELION -1, see module docstring), so
    without saying so this reads as a plain number that could be silently
    misread as ctfplotter's own native output."""
    h, ratio, x = parsed.get('relion_handedness'), parsed.get('ratio'), parsed.get('polarity_x')
    if h is None:
        return 'Could not determine tilt handedness from ctfplotter output.'
    x_note = f'RELION convention, converted from ctfplotter -testInv x={x:+d}' if x is not None else 'RELION convention'
    if not parsed.get('clear'):
        return (f'Result not clear enough (ratio {ratio} < {_CLEAR_RATIO}) -- '
                f'best guess is {h:+d} ({x_note}) but treat with caution.')
    return f'Import into RELION using {h:+d} tilt handedness.  ({x_note}; ratio {ratio})'


def _parse_testinv_stdout(text: str) -> dict:
    out = {
        'polarity_x': None, 'ratio': None, 'clear': False,
        'relion_handedness': None,
        'left_asgiven': None, 'right_asgiven': None, 'diff_asgiven': None,
        'left_inverted': None, 'right_inverted': None, 'diff_inverted': None,
    }
    m = _RE_POLARITY.search(text)
    if m:
        x = int(m.group(1))
        out['polarity_x'] = x
        out['ratio'] = float(m.group(2))
        out['clear'] = out['ratio'] >= _CLEAR_RATIO
        out['relion_handedness'] = _POLARITY_TO_RELION.get(x)
    m = _RE_LR_ASGIVEN.search(text)
    if m:
        out['left_asgiven'], out['right_asgiven'], out['diff_asgiven'] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))
    m = _RE_LR_INVERT.search(text)
    if m:
        out['left_inverted'], out['right_inverted'], out['diff_inverted'] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Running ctfplotter
# ─────────────────────────────────────────────────────────────────────────────

def _run_testinv(ctfplotter_bin, st_path, tlt_path, aangle, apix_A, scan_lo, scan_hi,
                  work_dir, env, kv, cs, ac, timeout):
    # tlt_path (-angleFn) is the IMOD-format .tlt file from _Imod/ -- its
    # values are identical to the .aln file's own TILT column (verified
    # directly). Whether that's raw nominal or already alpha-offset-
    # corrected depends on the source run's -TiltCor setting (confirmed
    # dataset-wide -- see CLAUDE.md's alpha_offset convention section and
    # _make_plot()'s docstring); the caller passes source_alpha_offset
    # through so results/plots can be labelled correctly either way. We
    # never add anything to it ourselves -- used exactly as-is.
    cmd = [
        ctfplotter_bin, '-volt', f'{kv:g}', '-cs', f'{cs:g}', '-ampContrast', f'{ac:g}',
        '-input', str(st_path), '-angleFn', str(tlt_path), '-aAngle', str(aangle),
        '-pixelSize', str(apix_A / 10.0),
        '-scan', f'{scan_lo:.0f},{scan_hi:.0f}', '-autoFit', '10,4',
        '-defFn', str(work_dir / 'testinv.defocus'), '-testInv', '0',
    ]
    env = dict(env)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env.pop('DISPLAY', None)
    ret = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True, timeout=timeout)
    return ret, cmd


def _build_half_stacks(st_path, xf_path, work_dir, newstack_bin, env, bin_factor):
    """Left/right ALIGNED half-stacks -- same construction DefocusGrad's own
    script uses (-xform to make tilt axis vertical, then -offset/-size to
    take each half), so the resulting plot is visually comparable to
    DefocusGrad's, not just similarly-styled."""
    with mrcfile.open(str(st_path), permissive=True, header_only=True) as mrc:
        nx, ny = int(mrc.header.nx), int(mrc.header.ny)
    outsize_x = nx // bin_factor // 2
    outsize_y = ny // bin_factor

    halves = {}
    for side, offset_x in (('left', -(nx // 4)), ('right', +(nx // 4))):
        out_path = work_dir / f'{side}.mrc'
        cmd = [newstack_bin, '-in', str(st_path), '-ou', str(out_path),
               '-xform', str(xf_path), '-offset', f'{offset_x},0', '-bin', str(bin_factor),
               '-size', f'{outsize_x},{outsize_y}', '-linear', '-antialias', '5']
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f'newstack ({side} half) failed: {ret.stderr.strip()}')
        halves[side] = out_path
    return halves


def _run_side_autofit(ctfplotter_bin, half_path, tlt_path, apix_A, bin_factor,
                       scan_lo, scan_hi, work_dir, env, kv, cs, ac, side, timeout):
    # Same tlt_path as -angleFn here, see _run_testinv above for what its
    # values actually are (depends on the source run's -TiltCor setting).
    def_path = work_dir / f'{side}.defocus'
    cmd = [
        ctfplotter_bin, '-volt', f'{kv:g}', '-cs', f'{cs:g}', '-ampContrast', f'{ac:g}',
        '-input', str(half_path), '-angleFn', str(tlt_path), '-aAngle', '0',
        '-pixelSize', str(apix_A * bin_factor / 10.0),
        '-scan', f'{scan_lo:.0f},{scan_hi:.0f}', '-autoFit', '0,0', '-save',
        '-defFn', str(def_path),
    ]
    env = dict(env)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env.pop('DISPLAY', None)
    ret = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True, timeout=timeout)
    if ret.returncode != 0 or not def_path.exists():
        raise RuntimeError(f'ctfplotter ({side} autofit) failed: {ret.stderr.strip()[-500:]}')

    tilts, defocus_A = [], []
    for line in def_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            ang1, ang2, defnm = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        tilts.append((ang1 + ang2) / 2.0)  # NOMINAL tilt (deg), not alpha-offset-corrected
        defocus_A.append(defnm * 10.0)  # nm -> A, matching DefocusGrad's plot units
    return np.array(tilts), np.array(defocus_A)


def _make_plot(tilts_l, def_l, tilts_r, def_r, out_path, tilt_is_corrected):
    """
    Top panel: DefocusGrad-style left/right defocus vs tilt, as before.
    Bottom panel: delta = right - left at matching tilts (same underlying
    .tlt file/view order for both halves, so index-paired -- not a
    tilt-angle match). Any common-mode trend shared by both sides (overall
    focus drift, ramp during acquisition -- anything that isn't the
    left-right geometric gradient itself) cancels out of the subtraction,
    which the raw per-side slopes above don't do: on real data (ts-083) the
    whole-frame defocus estimate ranged ~15000-34000 A across the tilt
    series (see shared/tilt_series_qc.py's defocus_tilt_spread_um), enough
    to swamp the smaller differential signal and make the two raw slopes
    read as same-sign ("inconclusive") even when the delta and -testInv
    both agree on a clear handedness.

    NOMINAL vs ALPHA-OFFSET-CORRECTED TILT -- resolved, read before changing
    this. tilts_l/tilts_r come straight from tlt_path (the IMOD _st.tlt
    file), and whether THAT is raw nominal or already specimen-referenced
    depends entirely on how the source AreTomo3 run was processed:
    confirmed dataset-wide (172/172 TS of a real project, comparing the
    same TS processed with -TiltCor 0 vs -TiltCor 1) that -TiltCor 1 bakes
    AlphaOffset directly into both the .aln TILT column and the IMOD
    _st.tlt derived from it -- every TILT value in the TiltCor=1 run was
    shifted from the TiltCor=0 run (itself confirmed to match the source
    mdoc's raw TiltAngle exactly) by precisely that TS's own AlphaOffset.
    See CLAUDE.md's alpha_offset convention section for the full test.
    tilt_is_corrected (source_alpha_offset != 0.0, read from aln_dir's own
    .aln -- NOT --analysis's alignment_data.json, which can be a different
    run processed differently) says which case applies for THIS plot, and
    the x-axis label reflects it accordingly. We never add alpha_offset
    ourselves either way -- the tlt_path value is used exactly as-is.

    This also resolves what was an open question during development: on
    the real data used above (ts-136, ts-074), the delta plot's zero-
    crossing landed near nominal tilt = 0 rather than near -AlphaOffset,
    which was confusing until this was understood -- the .tlt file used
    (from a -TiltCor 1 run) was already specimen-referenced, so 0 WAS the
    correct place for it to cross. A raw-nominal .tlt (-TiltCor 0) would be
    expected to cross nearer -AlphaOffset instead.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7.5))
    tilt_axis_label = ('Tilt angle (deg) -- alpha-offset corrected (-TiltCor baked in)'
                        if tilt_is_corrected else
                        'Nominal tilt angle (deg) -- not alpha-offset-corrected')

    slope_l = slope_r = None
    if len(tilts_l) >= 2:
        slope_l, intercept_l = np.polyfit(tilts_l, def_l, 1)
        ax1.scatter(tilts_l, def_l, color='blue', label='Defocus left-side', s=18)
        xs = np.linspace(tilts_l.min(), tilts_l.max(), 50)
        ax1.plot(xs, slope_l * xs + intercept_l, '--', color='blue', label='Linear fit left-side')
    if len(tilts_r) >= 2:
        slope_r, intercept_r = np.polyfit(tilts_r, def_r, 1)
        ax1.scatter(tilts_r, def_r, color='orange', label='Defocus right-side', s=18)
        xs = np.linspace(tilts_r.min(), tilts_r.max(), 50)
        ax1.plot(xs, slope_r * xs + intercept_r, '--', color='orange', label='Linear fit right-side')
    ax1.set_xlabel(tilt_axis_label)
    ax1.set_ylabel('Defocus (A)')
    ax1.legend(fontsize=8)

    # Cruder than -testInv's own verdict: -testInv compares defocus at one
    # strategically-chosen high-tilt view set, while this is a naive global
    # linear fit across the whole tilt range. Kept as a secondary/diagnostic
    # note in the report, never as the primary verdict -- that's always
    # -testInv's own result.
    slope_verdict = None
    if slope_l is not None and slope_r is not None:
        if slope_l < 0 and slope_r > 0:
            slope_verdict = -1
        elif slope_l > 0 and slope_r < 0:
            slope_verdict = 1
        else:
            slope_verdict = 0

    delta_slope = delta_corr = delta_verdict = None
    if len(tilts_l) == len(tilts_r) and len(tilts_l) >= 3 and np.allclose(tilts_l, tilts_r, atol=0.05):
        delta = def_r - def_l
        delta_slope, delta_intercept = np.polyfit(tilts_l, delta, 1)
        with np.errstate(invalid='ignore'):
            delta_corr = float(np.corrcoef(tilts_l, delta)[0, 1])
        # right - left increasing with tilt (delta_slope > 0) is the same
        # "right more underfocused at positive tilt" pattern -testInv's
        # x=1 (-> RELION -1) describes -- see module docstring.
        if delta_slope > 0:
            delta_verdict = -1
        elif delta_slope < 0:
            delta_verdict = 1
        else:
            delta_verdict = 0
        ax2.scatter(tilts_l, delta, color='green', s=18, label='Delta (right - left)')
        xs = np.linspace(tilts_l.min(), tilts_l.max(), 50)
        ax2.plot(xs, delta_slope * xs + delta_intercept, '--', color='green', label='Linear fit')
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.set_xlabel(tilt_axis_label)
        ax2.set_ylabel('Defocus delta, right - left (A)')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'delta unavailable (mismatched tilt sampling)',
                  ha='center', va='center', transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return slope_l, slope_r, slope_verdict, delta_slope, delta_corr, delta_verdict


# ─────────────────────────────────────────────────────────────────────────────
# Per-TS orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_ts(ts_name, st_path, xf_path, tlt_path, out_dir, args,
                 ctfplotter_bin, newstack_bin, env, source_alpha_offset, aangle, ts_data=None):
    ts_out = out_dir / ts_name
    ts_out.mkdir(parents=True, exist_ok=True)
    result = {'ts': ts_name, 'status': 'ok', 'error': None}

    apix_A = _apix_A(st_path)
    scan_lo, scan_hi, center_um = _scan_range_nm(ts_data or {}, args.scan_pad_um)
    result['scan_range_um'] = [round(scan_lo / 1000, 2), round(scan_hi / 1000, 2)]
    result['known_mean_defocus_um'] = round(center_um, 2)
    result['aangle_deg'] = round(aangle, 2)
    result['source_alpha_offset'] = round(source_alpha_offset, 3)
    # Whether tlt_path already has AlphaOffset baked in -- confirmed
    # dataset-wide that -TiltCor 1 bakes it into both .aln's TILT column and
    # the IMOD _st.tlt derived from it, while -TiltCor 0 leaves both raw
    # nominal (and AlphaOffset reads 0.0) -- see CLAUDE.md. Used purely for
    # labelling the plot axis correctly; never added to any tilt value here.
    result['tilt_is_corrected'] = source_alpha_offset != 0.0

    if args.dry_run:
        print(f'[dry-run] {ts_name}: -testInv aAngle={aangle:.2f} '
              f'scan={scan_lo:.0f},{scan_hi:.0f}nm  st={st_path} tlt={tlt_path}')
        if not args.skip_plots:
            print(f'[dry-run] {ts_name}: would build left/right half-stacks from {xf_path}, bin={args.bin}')
        return result

    try:
        ret, cmd = _run_testinv(ctfplotter_bin, st_path, tlt_path, aangle, apix_A,
                                 scan_lo, scan_hi, ts_out, env, args.kv, args.cs,
                                 args.amp_contrast, args.timeout)
        parsed = _parse_testinv_stdout(ret.stdout)
        result.update(parsed)
        (ts_out / 'testinv.log').write_text(
            'CMD: ' + ' '.join(cmd) + '\n\n--- stdout ---\n' + ret.stdout + '\n--- stderr ---\n' + ret.stderr
            + '\n--- CONCLUSION ---\n' + _relion_verdict_line(parsed) + '\n')
        if parsed['polarity_x'] is None:
            result['status'] = 'warn'
            result['error'] = 'could not parse "Angle polarity needed" line from ctfplotter stdout'
    except subprocess.TimeoutExpired:
        result.update(status='error', error=f'ctfplotter -testInv timed out after {args.timeout}s')
        return result
    except Exception as e:
        result.update(status='error', error=f'-testInv failed: {e}')
        return result

    if not args.skip_plots:
        try:
            halves = _build_half_stacks(st_path, xf_path, ts_out, newstack_bin, env, args.bin)
            tilts_l, def_l = _run_side_autofit(
                ctfplotter_bin, halves['left'], tlt_path, apix_A, args.bin,
                scan_lo, scan_hi, ts_out, env, args.kv, args.cs, args.amp_contrast, 'left', args.timeout)
            tilts_r, def_r = _run_side_autofit(
                ctfplotter_bin, halves['right'], tlt_path, apix_A, args.bin,
                scan_lo, scan_hi, ts_out, env, args.kv, args.cs, args.amp_contrast, 'right', args.timeout)
            plot_path = ts_out / f'{ts_name}_ctfplotter.png'
            slope_l, slope_r, slope_verdict, delta_slope, delta_corr, delta_verdict = _make_plot(
                tilts_l, def_l, tilts_r, def_r, plot_path, result['tilt_is_corrected'])
            result['plot_path'] = str(plot_path)
            result['slope_left'] = None if slope_l is None else round(float(slope_l), 2)
            result['slope_right'] = None if slope_r is None else round(float(slope_r), 2)
            result['slope_relion_handedness'] = slope_verdict
            result['delta_slope'] = None if delta_slope is None else round(float(delta_slope), 2)
            result['delta_corr'] = None if delta_corr is None else round(float(delta_corr), 3)
            result['delta_relion_handedness'] = delta_verdict
        except subprocess.TimeoutExpired:
            result['plot_error'] = f'plot generation timed out after {args.timeout}s'
        except Exception as e:
            result['plot_error'] = f'plot generation failed: {e}'
        finally:
            for name in ('left.mrc', 'right.mrc'):
                p = ts_out / name
                if p.exists():
                    p.unlink()

    return result


def _consensus(per_ts: dict) -> str:
    conclusive = [r['relion_handedness'] for r in per_ts.values()
                  if r.get('status') == 'ok' and r.get('clear') and r.get('relion_handedness') is not None]
    n_excluded_ambiguous = sum(1 for r in per_ts.values()
                                if r.get('status') == 'ok' and r.get('relion_handedness') is not None
                                and not r.get('clear'))
    suffix = f' [{n_excluded_ambiguous} more not-clear-enough result(s) excluded]' if n_excluded_ambiguous else ''
    if not conclusive:
        return 'inconclusive' + suffix
    counts = Counter(conclusive)
    if len(counts) == 1:
        val = conclusive[0]
        label = f'{val:+d}' if len(conclusive) > 1 else f'{val:+d} (from only 1 TS)'
        return label + suffix
    return f'inconsistent ({dict(counts)})' + suffix


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s):
    import html
    return html.escape(str(s))


def _test_summary_html(r: dict) -> str:
    """Side-by-side summary of the two independent tests this command runs
    for one TS: the built-in ctfplotter -testInv result (primary -- this is
    the value used for the overall consensus and what project-summary
    shows) and the left/right test (secondary/diagnostic -- built in-house
    from half-stacks, cross-checked two ways: raw per-side slopes and the
    right-left delta)."""
    h, ratio, clear = r.get('relion_handedness'), r.get('ratio'), r.get('clear')
    if h is None:
        builtin_txt = 'unparseable'
    else:
        builtin_txt = f'{h:+d}  (ratio {ratio}{"" if clear else ", not clear"})'

    lr_bits = []
    if r.get('delta_relion_handedness') is not None:
        lr_bits.append(_esc(f'delta {r["delta_relion_handedness"]:+d} (corr {r.get("delta_corr")})'))
    if r.get('slope_relion_handedness') is not None:
        lr_bits.append(_esc(f'slopes {r["slope_relion_handedness"]:+d}'))
    lr_txt = ' · '.join(lr_bits) if lr_bits else _esc('not run (--skip-plots or failed)')

    return (f'<div class="test-summary">'
            f'<div><span class="test-label">Built-in test (-testInv), RELION convention:</span> {_esc(builtin_txt)}</div>'
            f'<div><span class="test-label">Left/right test (in-house), RELION convention:</span> {lr_txt}</div>'
            f'</div>')


def _make_html(per_ts: dict, selection_scores: dict, consensus: str, out_path: Path):
    cards = []
    for ts_name, r in per_ts.items():
        sel = selection_scores.get(ts_name, {})
        status = r.get('status')
        h = r.get('relion_handedness')
        if status != 'ok':
            badge_cls, badge_txt = 'err', f'ERROR: {_esc(r.get("error") or "unknown")}'
        elif h is None:
            badge_cls, badge_txt = 'err', 'unparseable'
        elif not r.get('clear'):
            badge_cls, badge_txt = 'warn', f'RELION {h:+d} (not clear, ratio {r.get("ratio")})'
        else:
            badge_cls, badge_txt = 'ok', f'RELION {h:+d}  (ratio {r.get("ratio")})'
        test_summary = _test_summary_html(r) if status == 'ok' else ''

        plot_html = ''
        if r.get('plot_path'):
            import base64
            try:
                b64 = base64.b64encode(Path(r['plot_path']).read_bytes()).decode()
                slope_note = (f"slope L={r.get('slope_left')} R={r.get('slope_right')} "
                               f"→ {r.get('slope_relion_handedness')}"
                               if r.get('slope_relion_handedness') is not None else '')
                delta_note = (f"delta (right-left) slope={r.get('delta_slope')} A/deg, "
                               f"corr={r.get('delta_corr')} → {r.get('delta_relion_handedness')}"
                               if r.get('delta_relion_handedness') is not None else '')
                plot_html = (f'<img src="data:image/png;base64,{b64}">'
                              f'<div class="note">{_esc(delta_note)}</div>'
                              f'<div class="note">{_esc(slope_note)}</div>')
            except OSError:
                pass
        elif r.get('plot_error'):
            plot_html = f'<div class="note err">{_esc(r["plot_error"])}</div>'

        lr_html = ''
        if r.get('left_asgiven') is not None:
            lr_html = (f'<div class="note">As given: left {r["left_asgiven"]:.2f} right '
                       f'{r["right_asgiven"]:.2f} (diff {r["diff_asgiven"]:.3f}) &middot; '
                       f'Inverted: left {r["left_inverted"]:.2f} right {r["right_inverted"]:.2f} '
                       f'(diff {r["diff_inverted"]:.3f})</div>')

        cards.append(f"""
      <div class="card">
        <div class="card-title">{_esc(ts_name)} <span class="badge {badge_cls}">{badge_txt}</span></div>
        {test_summary}
        <div class="note">coverage {sel.get('coverage_deg', '?')}&deg; &middot; high-tilt CTF res
          {sel.get('high_tilt_ctf_res_A', '?')}&Aring; &middot; known defocus
          {r.get('known_mean_defocus_um', '?')}&micro;m &middot; aAngle {r.get('aangle_deg', '?')}&deg;
          &middot; tilt is {'alpha-offset corrected' if r.get('tilt_is_corrected') else 'raw nominal'}
          (source AlphaOffset {r.get('source_alpha_offset', '?')}&deg;)</div>
        {lr_html}
        {plot_html}
      </div>""")

    consensus_cls = 'warn' if ('inconsist' in consensus or 'inconclus' in consensus) else 'ok'
    about_html = """
  <div class="about">
    <p><b>Two independent tests are run per TS.</b> The <b>built-in test</b>
    (IMOD ctfplotter's own <code>-testInv</code>) is the primary result --
    it's what the Consensus above and project-summary's "defocus handedness"
    both use. The <b>left/right test</b> is a second, in-house cross-check:
    the raw tilt series is split into left/right halves (rotated so the
    tilt axis is vertical, matching the DefocusGrad convention) and each
    half's defocus is fit separately across the whole tilt range, so it can
    be plotted and inspected visually rather than trusted as a single
    number.</p>
    <p><b>TS are auto-selected</b> for wide tilt coverage, good high-tilt
    CTF resolution, and -- primarily -- a <b>flat</b> AreTomo3-reported
    defocus-vs-tilt trend (low spread in the overall, non-split per-frame
    defocus estimate). A flat TS is less likely to have a confounding
    trend (focus drift, poor eucentricity, etc.) on top of the left-right
    signal, which makes both tests -- and especially the left/right delta
    below -- easier to read cleanly.</p>
    <p><b>Reading the plots -- what + and - tilt mean:</b> the x-axis label
    on each plot says whether it's showing <b>nominal (stage) tilt</b> or
    <b>alpha-offset-corrected tilt</b> -- this depends on whether the source
    AreTomo3 run used <code>-TiltCor</code>: confirmed dataset-wide that
    <code>-TiltCor 1</code> bakes the AlphaOffset correction directly into
    the tilt values (both the .aln TILT column and the IMOD _st.tlt file
    used here), while <code>-TiltCor 0</code> leaves them raw. Left = blue,
    right = orange either way. If the tilt handedness convention is
    <b>RELION -1</b>, the LEFT side's defocus should <b>decrease</b> towards
    positive tilt angles (a negative slope) while the RIGHT side's defocus
    <b>increases</b> towards positive tilt (a positive slope) -- an "X"
    shape. The opposite pattern (left increasing, right decreasing) means
    <b>RELION +1</b>. Same-sign slopes on both sides means the two halves
    aren't giving a clear signal either way.</p>
    <p><b>The delta plot</b> (bottom panel) shows right-side defocus minus
    left-side defocus at each matching tilt angle. Subtracting removes any
    trend the two sides share in common (e.g. focus drift over the course
    of acquisition, unrelated to handedness), leaving just the left-right
    difference that handedness actually predicts: it should trend from
    negative (right &lt; left) to positive (right &gt; left) as tilt
    increases for RELION -1, and the opposite for RELION +1, crossing zero
    at whatever tilt the specimen itself is flat -- at nominal tilt = 0 if
    the plot is already alpha-offset-corrected, or at nominal tilt =
    -AlphaOffset if it's still raw nominal. A flatter, noisier delta line
    without a clear trend means this TS isn't giving a clean signal. See
    <code>ctf_handedness.py</code>'s <code>_make_plot()</code> docstring and
    CLAUDE.md's alpha_offset convention section for the dataset-wide test
    behind this.</p>
  </div>"""
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ctf-handedness QC</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #fff; color: #263238;
          padding: 32px 16px; }}
  h1 {{ color: #0d47a1; margin-bottom: 6px; }}
  .consensus {{ font-size: 1.1em; margin-bottom: 16px; }}
  .consensus.ok {{ color: #2e7d32; }}
  .consensus.warn {{ color: #b26a00; }}
  .about {{
    background: #f0f4f8; border: 1px solid #dbe3ea; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 24px; max-width: 900px;
  }}
  .about p {{ font-size: 0.86em; color: #37474f; margin: 0 0 10px; line-height: 1.5; }}
  .about p:last-child {{ margin-bottom: 0; }}
  .about code {{ background: #e0e6ea; padding: 1px 5px; border-radius: 4px; font-size: 0.95em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
  .card {{ background: #f5f7fa; border: 1px solid #e0e6ea; border-radius: 10px; padding: 16px; }}
  .card-title {{ font-weight: 600; margin-bottom: 6px; }}
  .badge {{ font-size: 0.8em; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }}
  .badge.ok {{ background: #c8e6c9; color: #1b5e20; }}
  .badge.warn {{ background: #ffe0b2; color: #8d5300; }}
  .badge.err {{ background: #ffcdd2; color: #b71c1c; }}
  .note {{ font-size: 0.8em; color: #607d8b; margin: 6px 0; }}
  .note.err {{ color: #b71c1c; }}
  .test-summary {{
    background: #ffffff; border: 1px solid #e0e6ea; border-radius: 6px;
    padding: 6px 10px; margin: 8px 0; font-size: 0.82em;
  }}
  .test-label {{ color: #78909c; }}
  img {{ max-width: 100%; border-radius: 6px; }}
</style>
</head>
<body>
  <h1>ctf-handedness QC (IMOD ctfplotter -testInv)</h1>
  <div class="consensus {consensus_cls}">Consensus (built-in test, RELION convention): <b>{_esc(consensus)}</b></div>
  {about_html}
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    out_path.write_text(html_out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'ctf-handedness',
        help='Defocus-handedness QC via IMOD ctfplotter -testInv (cross-check for defocusgrad)',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--analysis', '-A', help='analyse output dir (has alignment_data.json); default: latest recorded')
    p.add_argument('--aln-dir', help='dir with ts-XXX_Imod/ subdirs (analyse\'s own --input); default from project.json')
    p.add_argument('--cmd0-dir', help='run-aretomo3 --cmd 0 output dir (has ts-XXX.mrc); default from project.json')
    p.add_argument('--output', '-o', default='ctf_handedness_qc', help='output directory (default: ctf_handedness_qc)')

    p.add_argument('--n-ts', type=int, default=4, help='number of TS to auto-select (default: 4)')
    p.add_argument('--select-ts', nargs='+', help='explicit TS names instead of auto-selection')

    p.add_argument('--dry-run', action='store_true', help='print planned commands without running them')
    p.add_argument('--skip-plots', action='store_true', help='skip left/right half-stack plot generation (testInv only, faster)')
    p.add_argument('--timeout', type=int, default=1800, help='per-subprocess timeout in seconds (default: 1800)')
    p.add_argument('--jobs', '-j', type=int, default=0, help='parallel TS workers (default: one per TS)')

    p.add_argument('--bin', type=int, default=1, help='binning for left/right half-stacks used in plots (default: 1)')
    p.add_argument('--scan-pad-um', type=float, default=1.5,
                    help='+/- range (um) around each TS\'s known mean defocus for the ctfplotter -scan range (default: 1.5)')

    ctf = p.add_argument_group('CTF parameters (default from recorded run-aretomo3 params)')
    ctf.add_argument('--kv', type=float, help='voltage (kV)')
    ctf.add_argument('--cs', type=float, help='spherical aberration (mm)')
    ctf.add_argument('--amp-contrast', type=float, help='amplitude contrast')

    tools = p.add_argument_group('tool paths')
    tools.add_argument('--imod-dir', help=f'IMOD install dir (default: {_DEFAULT_IMOD_DIR})')

    p.set_defaults(func=run)


def run(args):
    proj = load_project()

    analysis_dir = Path(args.analysis) if args.analysis else get_latest_analysis_dir()
    if not analysis_dir:
        sys.exit('ctf-handedness: no --analysis given and none recorded in project.json; run `analyse` first')
    alignment_json = Path(analysis_dir) / 'alignment_data.json'
    if not alignment_json.exists():
        sys.exit(f'ctf-handedness: {alignment_json} not found')
    alignment_data = json.loads(alignment_json.read_text())

    aln_dir = args.aln_dir or proj.get('analyse', {}).get('args', {}).get('input')
    if not aln_dir:
        sys.exit('ctf-handedness: --aln-dir not given and analyse.args.input not recorded in project.json')
    aln_dir = Path(aln_dir)

    cmd0_dir = Path(args.cmd0_dir) if args.cmd0_dir else get_cmd0_outdir()
    if not cmd0_dir:
        sys.exit('ctf-handedness: --cmd0-dir not given and none recorded in project.json')

    imod_dir = args.imod_dir or resolve_tool_path('imod', _DEFAULT_IMOD_DIR)
    newstack_bin = str(Path(imod_dir) / 'bin' / 'newstack')
    ctfplotter_bin = str(Path(imod_dir) / 'bin' / 'ctfplotter')
    if not Path(newstack_bin).exists():
        sys.exit(f'ctf-handedness: newstack not found at {newstack_bin} (--imod-dir)')
    if not Path(ctfplotter_bin).exists():
        sys.exit(f'ctf-handedness: ctfplotter not found at {ctfplotter_bin} (--imod-dir)')

    run_params = get_run_params() or {}
    kv = args.kv if args.kv is not None else run_params.get('kv', 300.0)
    cs = args.cs if args.cs is not None else run_params.get('cs', 2.7)
    ac = args.amp_contrast if args.amp_contrast is not None else run_params.get('amp_contrast', 0.07)
    args.kv, args.cs, args.amp_contrast = kv, cs, ac

    if args.select_ts:
        selected = [(name, {}) for name in args.select_ts if name in alignment_data]
        missing = set(args.select_ts) - {n for n, _ in selected}
        if missing:
            sys.exit(f'ctf-handedness: TS not found in alignment_data.json: {sorted(missing)}')
    else:
        selected = _select_ts(alignment_data, args.n_ts)
    if not selected:
        sys.exit('ctf-handedness: no TS survived selection')
    selection_scores = {name: score for name, score in selected}

    out_dir = Path(args.output).resolve()
    if not args.dry_run:
        check_output_dir(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    env = _imod_env(newstack_bin, None)

    jobs = []
    for ts_name, _ in selected:
        st_path = Path(cmd0_dir) / f'{ts_name}.mrc'
        imod_ts_dir = aln_dir / f'{ts_name}_Imod'
        xf_path = imod_ts_dir / f'{ts_name}_st.xf'
        tlt_path = imod_ts_dir / f'{ts_name}_st.tlt'
        # AlphaOffset read from THIS run's own .aln (not alignment_data.json,
        # which may come from a different --analysis run processed with a
        # different -TiltCor setting, see CLAUDE.md's alpha_offset
        # convention section) -- tells us whether this .tlt already has the
        # correction baked in (-TiltCor 1, confirmed dataset-wide) or is raw
        # nominal (-TiltCor 0), so the plot axis can be labelled correctly.
        aln_source_path = aln_dir / f'{ts_name}.aln'
        missing = [str(p) for p in (st_path, tlt_path, aln_source_path) if not p.exists()]
        if not args.skip_plots and not xf_path.exists():
            missing.append(str(xf_path))
        if missing:
            print(f'SKIP {ts_name}: missing input(s): {missing}')
            continue
        # Parsed once from aln_dir's own .aln (NOT --analysis's
        # alignment_data.json, which can be a different AreTomo3 run --
        # e.g. cmd0 vs cmd1 -- with a different tilt-axis rotation; using
        # the wrong run's aAngle would subtly bias the left/right split).
        aln_source = parse_aln_file(aln_source_path)
        source_alpha_offset = aln_source.get('alpha_offset') or 0.0
        source_aangle = aln_source['frames'][0]['rot'] if aln_source.get('frames') else None
        if source_aangle is None:
            print(f'SKIP {ts_name}: no frames in {aln_source_path}')
            continue
        jobs.append((ts_name, st_path, xf_path, tlt_path, source_alpha_offset, source_aangle))

    if not jobs:
        sys.exit('ctf-handedness: no selected TS had all required input files')

    if args.dry_run:
        for ts_name, st_path, xf_path, tlt_path, source_alpha_offset, source_aangle in jobs:
            _run_one_ts(ts_name, st_path, xf_path, tlt_path, out_dir, args,
                        ctfplotter_bin, newstack_bin, env, source_alpha_offset, source_aangle,
                        ts_data=alignment_data.get(ts_name))
        return

    per_ts = {}
    max_workers = args.jobs if args.jobs > 0 else max(1, len(jobs))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_ts, ts_name, st_path, xf_path, tlt_path, out_dir, args,
                        ctfplotter_bin, newstack_bin, env, source_alpha_offset, source_aangle,
                        ts_data=alignment_data.get(ts_name)): ts_name
            for ts_name, st_path, xf_path, tlt_path, source_alpha_offset, source_aangle in jobs
        }
        for fut in as_completed(futures):
            ts_name = futures[fut]
            r = fut.result()
            per_ts[ts_name] = r
            if r.get('status') == 'ok' and r.get('relion_handedness') is not None:
                print(f'{ts_name}: {_relion_verdict_line(r)}')
            else:
                print(f'{ts_name}: {r.get("status")} — {r.get("error")}')

    consensus = _consensus(per_ts)
    print(f'\nConsensus (RELION convention): {consensus}')

    html_path = out_dir / 'index.html'
    _make_html(per_ts, selection_scores, consensus, html_path)

    update_section('ctf_handedness', {
        'command': 'ctf-handedness',
        'args': args_to_dict(args),
        'output_dir': str(out_dir),
        'consensus': consensus,
        'per_ts': {ts: {**r, 'selection': selection_scores.get(ts, {})} for ts, r in per_ts.items()},
    })
    record_analysis_run('ctf_handedness', str(out_dir))
    write_landing_page(Path.cwd())
    print(f'Report: {html_path}')
