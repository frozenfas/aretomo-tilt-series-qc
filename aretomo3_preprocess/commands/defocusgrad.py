"""
defocusgrad — automated defocus-handedness QC via DefocusGrad.

Wraps the DefocusGrad tool (CellArchLab/cryoet-scripts, based on Scaramuzza &
Castaño-Diez's TomographyTools) to empirically verify the defocus-handedness
convention (rlnTomoHand, ±1) that relion5-convert currently just passes
through from AreTomo3's own _CTF.txt `dfhand` column (which reflects
whatever handedness AreTomo3 assumed going in, not an independent
measurement).

DefocusGrad splits an aligned tilt series into left/right halves, runs
CTFFIND on each half separately, and fits defocus vs. tilt-angle slopes --
the sign combination of the two slopes gives the handedness. It needs, per
TS: the *unaligned* stack (--st), and an IMOD-format .xf/.tlt pair (--xf/
--tlt). This pipeline already produces exactly those:
  - --st  : <cmd0_outdir>/ts-XXX.mrc          (run-aretomo3 --cmd 0 output)
  - --xf  : <aln_dir>/ts-XXX_Imod/ts-XXX_st.xf
  - --tlt : <aln_dir>/ts-XXX_Imod/ts-XXX_st.tlt
(the _Imod directories exist because run-aretomo3 runs with --out-imod 1
by default. aln_dir is analyse's own --input -- NOT --analysis/--output
below, which is a separate directory in general; defaults from analyse's
own recorded args in project.json, see --aln-dir.)

Runs on a handful of tilt series (--n-ts, default 4), NOT the whole
dataset -- auto-selected (not random) by reading alignment_data.json
directly (same sanctioned pattern as select-ts/trim-ts) for TS with wide
angular coverage and good CTF fit quality, especially at high tilts (the
frames DefocusGrad's own per-half CTF fits depend on most).

Requires `newstack` (IMOD) and `ctffind` (CTFFIND4/5) resolvable -- see
--imod-dir/--ctffind-bin/--defocusgrad-bin, remembered in project.json's
tool_paths section like other external-tool wrappers in this repo.

Example
-------
  aretomo3-preprocess defocusgrad \\
      --analysis run002-cmd1/analyse --n-ts 4 --output defocusgrad_qc
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

from aretomo3_preprocess.shared.project_json import load as load_project, update_section, args_to_dict
from aretomo3_preprocess.shared.project_state import (
    get_latest_analysis_dir, get_cmd0_outdir, get_run_params,
    resolve_tool_path, record_analysis_run,
)
from aretomo3_preprocess.shared.landing_page import write_landing_page
from aretomo3_preprocess.shared.output_guard import check_output_dir

_DEFAULT_IMOD_DIR       = '/opt/IMOD'
_DEFAULT_CTFFIND5_BIN   = '/opt/ctffind5/cisTEM/bin/ctffind'
_DEFAULT_CTFFIND4_BIN   = '/opt/ctffind/bin/ctffind'
_DEFAULT_DEFOCUSGRAD_BIN = str(Path.home() / 'local' / 'cryoet-scripts' / 'defocusgrad' / 'defocusgrad')


# ─────────────────────────────────────────────────────────────────────────────
# TS auto-selection
# ─────────────────────────────────────────────────────────────────────────────

def _score_tilt_series(alignment_data: dict) -> dict:
    """
    {ts_name: {'coverage_deg', 'high_tilt_ctf_res', 'overall_ctf_res'}} for
    every real TS in an already-loaded alignment_data.json -- coverage is
    the angular range of retained (non-dark) frames; CTF resolution figures
    are mean fit_spacing_A (lower = better), 'high tilt' meaning the top
    tercile of |tilt| for that TS specifically (not a fixed degree cutoff,
    since different TS cover different ranges).
    """
    scores = {}
    for ts_name, data in alignment_data.items():
        frames = [f for f in data.get('frames', []) if f.get('tilt') is not None]
        if len(frames) < 3:
            continue
        tilts = [f['tilt'] for f in frames]
        coverage_deg = max(tilts) - min(tilts)

        res_all = [f['fit_spacing_A'] for f in frames if f.get('fit_spacing_A') is not None]
        if not res_all:
            continue

        abs_tilts_sorted = sorted(abs(t) for t in tilts)
        high_tilt_cutoff = abs_tilts_sorted[int(len(abs_tilts_sorted) * 2 / 3)]
        res_high = [f['fit_spacing_A'] for f in frames
                    if f.get('fit_spacing_A') is not None and abs(f['tilt']) >= high_tilt_cutoff]

        scores[ts_name] = {
            'coverage_deg':      round(coverage_deg, 1),
            'overall_ctf_res_A': round(float(np.mean(res_all)), 2),
            'high_tilt_ctf_res_A': round(float(np.mean(res_high)), 2) if res_high else round(float(np.mean(res_all)), 2),
        }
    return scores


def _select_ts(alignment_data: dict, n_ts: int, coverage_pctile: float = 75.0) -> list:
    """
    Pick n_ts TS: keep those with coverage >= coverage_pctile among all TS
    (the "high tilt coverage" qualifying filter), then rank survivors by
    high-tilt CTF resolution ascending (best first), overall CTF resolution
    as tie-breaker. Returns a list of (ts_name, score_dict), best first.
    """
    scores = _score_tilt_series(alignment_data)
    if not scores:
        return []
    coverages = [s['coverage_deg'] for s in scores.values()]
    coverage_thresh = float(np.percentile(coverages, coverage_pctile))
    survivors = [(name, s) for name, s in scores.items() if s['coverage_deg'] >= coverage_thresh]
    survivors.sort(key=lambda t: (t[1]['high_tilt_ctf_res_A'], t[1]['overall_ctf_res_A']))
    return survivors[:n_ts]


# ─────────────────────────────────────────────────────────────────────────────
# Running DefocusGrad
# ─────────────────────────────────────────────────────────────────────────────

def _imod_env(newstack_bin: str, ctffind_bin: str) -> dict:
    """
    Env for the defocusgrad subprocess: it shells out to `newstack`/`ctffind`
    by bare name, so both binaries' directories need to be on PATH, and
    IMOD_DIR needs to be set for newstack's own wrapper script to find its
    "realbin" -- same pattern as pytom_ribo_auto.py's _resample_volume().

    MPLBACKEND=Agg is also required: the defocusgrad script calls
    plt.show() unconditionally after plt.savefig() with no backend
    override of its own -- without this, matplotlib picks an interactive
    backend and plt.show() hangs forever waiting for a GUI that can never
    appear in a subprocess (confirmed: it genuinely hung, not just slow).
    """
    imod_dir = str(Path(newstack_bin).resolve().parent.parent)  # .../bin/newstack -> ...
    env = dict(os.environ)
    env['IMOD_DIR'] = os.environ.get('IMOD_DIR', imod_dir)
    env['PATH'] = f"{Path(newstack_bin).parent}:{Path(ctffind_bin).parent}:{env.get('PATH', '')}"
    env['MPLBACKEND'] = 'Agg'
    return env


_RE_HANDEDNESS   = re.compile(r'Import into RELION using ([+-]1) tilt handedness')
_RE_INCONCLUSIVE = re.compile(r'results inconclusive')
_RE_CORR_LEFT    = re.compile(r'Correlation coefficient for left-side fit:\s*([\d.]+)')
_RE_CORR_RIGHT   = re.compile(r'Correlation coefficient for right-side fit:\s*([\d.]+)')
_RE_SLOPE_LEFT   = re.compile(r'Slope for left-side fit:\s*(-?[\d.]+)')
_RE_SLOPE_RIGHT  = re.compile(r'Slope for right-side fit:\s*(-?[\d.]+)')


def _parse_defocusgrad_stdout(text: str) -> dict:
    """Extract the handedness verdict + fit-quality numbers DefocusGrad
    already computes and prints, rather than re-deriving them ourselves."""
    m = _RE_HANDEDNESS.search(text)
    handedness = int(m.group(1)) if m else (0 if _RE_INCONCLUSIVE.search(text) else None)
    corr_left  = float(_RE_CORR_LEFT.search(text).group(1))  if _RE_CORR_LEFT.search(text)  else None
    corr_right = float(_RE_CORR_RIGHT.search(text).group(1)) if _RE_CORR_RIGHT.search(text) else None
    slope_left  = float(_RE_SLOPE_LEFT.search(text).group(1))  if _RE_SLOPE_LEFT.search(text)  else None
    slope_right = float(_RE_SLOPE_RIGHT.search(text).group(1)) if _RE_SLOPE_RIGHT.search(text) else None
    reliable = (corr_left is not None and corr_right is not None
                and corr_left > 0.5 and corr_right > 0.5)
    return {
        'handedness':  handedness,
        'corr_left':   corr_left,
        'corr_right':  corr_right,
        'slope_left':  slope_left,
        'slope_right': slope_right,
        'reliable':    reliable,
    }


def _read_ctf_diagnostics(ts_out: Path, strootname: str) -> dict:
    """
    CTFFIND's own per-half fit quality (cross-correlation score + fit
    resolution) -- independent of, and a useful cross-check against,
    DefocusGrad's own corr_left/corr_right (which measures how well
    defocus fits a straight line vs. tilt, not whether CTFFIND found a
    good CTF in the first place). Neither aligned stack's diagnostic .txt
    is deleted by defocusgrad's own cleanup (only the aligned .mrc stacks
    are, and only without --no_clean), so these are always available to
    read after a successful run.

    Columns (same format read by defocusgrad's own np.loadtxt call, and by
    RELION's ctffind_runner.cpp -- confirmed against that source): micro-
    graph#, defocus1, defocus2, astig_angle, phase_shift, score (cross-
    correlation, higher = better), fit resolution in Angstrom (lower =
    better; CTFFIND prints the literal string "inf" when no Thon rings
    were fit at all, treated here as a very poor 999 Å).
    """
    def _read_one(path):
        if not path.exists():
            return None
        scores, resolutions = [], []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                try:
                    scores.append(float(parts[5]))
                    res_s = parts[6]
                    resolutions.append(999.0 if res_s.lower() == 'inf' else float(res_s))
                except ValueError:
                    continue
        if not scores:
            return None
        return {'score_mean': round(float(np.mean(scores)), 4),
                'res_mean_A': round(float(np.mean(resolutions)), 2)}

    return {
        'ctf_left':  _read_one(ts_out / f'{strootname}_diagnostic_output_left.txt'),
        'ctf_right': _read_one(ts_out / f'{strootname}_diagnostic_output_right.txt'),
    }


def _run_one_ts(ts_name, st_path, xf_path, tlt_path, out_dir, args,
                defocusgrad_bin, newstack_bin, ctffind_bin):
    ts_out = out_dir / ts_name
    ts_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        defocusgrad_bin,
        '--st', str(st_path), '--xf', str(xf_path), '--tlt', str(tlt_path),
        '--bin', str(args.bin),
        '--kV', str(args.kv), '--Cs', str(args.cs), '--Ac', str(args.amp_contrast),
        '--exclude_negative', str(args.exclude_negative),
        '--exclude_positive', str(args.exclude_positive),
    ]
    if args.ctffind4:
        cmd.append('--ctffind4')

    print(f'\n[{ts_name}] + {" ".join(cmd)}')
    if args.dry_run:
        return {'handedness': None, 'dry_run': True}

    env = _imod_env(newstack_bin, ctffind_bin)
    log_path = ts_out / f'{ts_name}_defocusgrad.log'
    try:
        ret = subprocess.run(cmd, cwd=str(ts_out), env=env,
                             capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or '') + '\n--- stderr ---\n' + (exc.stderr or '')
                            + f'\n--- TIMED OUT after {args.timeout}s ---\n')
        print(f'  ERROR: defocusgrad timed out after {args.timeout}s for {ts_name} -- see {log_path}')
        return {'handedness': None, 'error': 'timeout', 'log': str(log_path)}

    log_path.write_text((ret.stdout or '') + '\n--- stderr ---\n' + (ret.stderr or ''))

    if ret.returncode != 0:
        print(f'  ERROR: defocusgrad exited {ret.returncode} for {ts_name} -- see {log_path}')
        return {'handedness': None, 'error': f'exit {ret.returncode}', 'log': str(log_path)}

    result = _parse_defocusgrad_stdout(ret.stdout)
    plot_name = f'{Path(st_path).stem}_defocusgrad.png'
    plot_path = ts_out / plot_name
    result['plot_path'] = str(plot_path) if plot_path.exists() else None
    result['log'] = str(log_path)
    result.update(_read_ctf_diagnostics(ts_out, Path(st_path).stem))
    if result['handedness'] is None and 'error' not in result:
        print(f'  WARNING: could not parse a handedness verdict from defocusgrad output for {ts_name}')
    for side in ('ctf_left', 'ctf_right'):
        d = result.get(side)
        if d is None:
            print(f'  WARNING: no CTFFIND diagnostic output found for {ts_name} ({side})')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

_CTF_POOR_RES_A = 15.0  # CTFFIND fit resolution worse than this = don't trust that side's fit


def _ctf_is_poor(r: dict) -> bool:
    """True if CTFFIND itself found a poor fit on either half -- a clean
    ±1 handedness verdict is only as trustworthy as the CTF estimation it
    was built on, and DefocusGrad's own corr_left/corr_right (how well
    defocus fits a straight line vs. tilt) doesn't catch this: a handful
    of noisy defocus points can still trace a passable line by chance."""
    ctf_left, ctf_right = r.get('ctf_left'), r.get('ctf_right')
    if ctf_left is None or ctf_right is None:
        return True
    return ctf_left['res_mean_A'] > _CTF_POOR_RES_A or ctf_right['res_mean_A'] > _CTF_POOR_RES_A


def _consensus(per_ts: dict) -> str:
    """
    Majority vote across TS with a clean ±1 verdict AND a good CTFFIND fit
    on both halves -- a TS with a poor CTF fit is excluded from the vote
    even if it happened to produce a ±1 verdict, since that verdict isn't
    trustworthy (see _ctf_is_poor). Reports how many were excluded this
    way so a consensus built on very few TS is visible as such, not
    silently presented with the same confidence as one built on many.
    """
    trusted = [r['handedness'] for r in per_ts.values()
              if r.get('handedness') in (-1, 1) and not _ctf_is_poor(r)]
    n_excluded_poor_ctf = sum(1 for r in per_ts.values()
                              if r.get('handedness') in (-1, 1) and _ctf_is_poor(r))
    suffix = f' [{n_excluded_poor_ctf} more ±1 verdict(s) excluded for a poor CTFFIND fit]' if n_excluded_poor_ctf else ''
    if not trusted:
        return 'inconclusive' + suffix
    counts = Counter(trusted)
    (top_val, top_n), = counts.most_common(1)
    if top_n == len(trusted):
        return f'{top_val}{suffix}' if len(trusted) > 1 else f'{top_val} (from only 1 TS){suffix}'
    return f'inconsistent ({dict(counts)}){suffix}'


def _make_html(per_ts: dict, selection_scores: dict, consensus: str, out_path: Path):
    import html as _html
    def esc(x):
        return _html.escape(str(x))

    cards = []
    for ts_name, r in per_ts.items():
        h = r.get('handedness')
        if h in (-1, 1):
            if _ctf_is_poor(r):
                badge_cls, badge_txt = 'warn', f'{h:+d} (poor CTF)'
            else:
                badge_cls, badge_txt = 'ok', f'{h:+d}'
        elif h == 0:
            badge_cls, badge_txt = 'warn', 'inconclusive'
        else:
            badge_cls, badge_txt = 'err', r.get('error', 'no result')
        img_html = ''
        if r.get('plot_path') and Path(r['plot_path']).exists():
            rel = Path(r['plot_path']).relative_to(out_path.parent)
            img_html = f'<img src="{esc(rel.as_posix())}" alt="{esc(ts_name)} defocusgrad plot">'
        else:
            img_html = '<div class="missing">no plot produced</div>'
        s = selection_scores.get(ts_name, {})
        ctf_left, ctf_right = r.get('ctf_left'), r.get('ctf_right')
        def _ctf_txt(d):
            return f'score={d["score_mean"]:.3f} res={d["res_mean_A"]:.1f}Å' if d else 'no CTFFIND output'
        ctf_poor = _ctf_is_poor(r)
        cards.append(f'''
      <div class="card">
        <div class="card-title">{esc(ts_name)} <span class="badge {badge_cls}">{esc(badge_txt)}</span></div>
        {img_html}
        <div class="stats">
          corr(left)={esc(r.get('corr_left'))}  corr(right)={esc(r.get('corr_right'))}
          {'<span class="warn">low correlation — not reliable</span>' if r.get('reliable') is False else ''}
        </div>
        <div class="stats">
          CTFFIND left: {esc(_ctf_txt(ctf_left))}  &middot;  CTFFIND right: {esc(_ctf_txt(ctf_right))}
          {'<span class="warn">poor CTF fit (&gt;15&Aring;) — treat this handedness verdict cautiously</span>' if ctf_poor else ''}
        </div>
        <div class="why">selected for: coverage={esc(s.get('coverage_deg'))}&deg;,
          high-tilt CTF res={esc(s.get('high_tilt_ctf_res_A'))}&Aring;,
          overall CTF res={esc(s.get('overall_ctf_res_A'))}&Aring;</div>
      </div>''')

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Defocus handedness QC (DefocusGrad)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #ffffff; color: #263238;
         padding: 32px 16px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #0d47a1; }}
  #consensus {{ font-size: 1.1em; padding: 10px 16px; border-radius: 8px;
                background: #f5f7fa; border: 1px solid #e0e6ea; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
  .card {{ background: #f5f7fa; border: 1px solid #e0e6ea; border-radius: 10px; padding: 14px; }}
  .card-title {{ font-weight: 600; margin-bottom: 8px; }}
  .card img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e6ea; }}
  .missing {{ color: #b0bec5; font-style: italic; padding: 40px 0; text-align: center; }}
  .badge {{ font-size: 0.8em; padding: 2px 8px; border-radius: 10px; font-weight: 600; }}
  .badge.ok {{ background: #c8e6c9; color: #1b5e20; }}
  .badge.warn {{ background: #ffe0b2; color: #8d6e00; }}
  .badge.err {{ background: #ffcdd2; color: #b71c1c; }}
  .stats {{ font-size: 0.85em; color: #546e7a; margin-top: 8px; }}
  .why {{ font-size: 0.78em; color: #90a4ae; margin-top: 6px; }}
  .warn {{ color: #b26a00; font-weight: 600; }}
</style>
</head>
<body>
  <h1>Defocus handedness QC (DefocusGrad)</h1>
  <div id="consensus">Dataset consensus: <b>{esc(consensus)}</b> &nbsp;
    (from {len(per_ts)} auto-selected tilt series — physical/mirror handedness is a
    separate check, see pytom-ribo-auto --check-handedness)</div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>"""
    out_path.write_text(html_out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_parser(subparsers):
    p = subparsers.add_parser(
        'defocusgrad',
        help='Empirically check defocus handedness (rlnTomoHand) via DefocusGrad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--analysis', '-A', default=None,
                   help='analyse output directory containing alignment_data.json '
                        '(default: last analyse run from project.json)')
    p.add_argument('--aln-dir', default=None,
                   help='AreTomo3 alignment output directory containing '
                        'ts-XXX_Imod/ts-XXX_st.{xf,tlt} -- this is analyse\'s '
                        'own --input, NOT its --output (they commonly differ). '
                        'Default: the --input that the --analysis run was '
                        'invoked with (project.json analyse.args.input).')
    p.add_argument('--cmd0-dir', default=None,
                   help='Directory with the unaligned ts-XXX.mrc stacks '
                        '(default: input_stacks.cmd0_outdir from project.json)')
    p.add_argument('--output', '-o', default='defocusgrad_qc',
                   help='Output directory')
    p.add_argument('--n-ts', type=int, default=4,
                   help='Number of tilt series to auto-select and check')
    p.add_argument('--select-ts', nargs='+', default=None, metavar='TS_NAME',
                   help='Explicit TS names to use instead of auto-selection')
    p.add_argument('--dry-run', action='store_true',
                   help='Print the selection and commands without running defocusgrad')
    p.add_argument('--clean', action='store_true',
                   help='Remove existing output directory before running')
    p.add_argument('--timeout', type=int, default=1800,
                   help='Per-TS timeout in seconds for the defocusgrad subprocess')
    p.add_argument('--jobs', '-j', type=int, default=0,
                   help='TS to process in parallel (default: 0 = one worker per '
                        'selected TS, i.e. fully parallel; each defocusgrad '
                        'invocation is itself CPU-light aside from CTFFIND, so '
                        'this scales fine on a multi-core machine)')

    ctf = p.add_argument_group('CTFFIND options (default from run_aretomo3_params in project.json)')
    ctf.add_argument('--kv', type=float, default=None, help='Acceleration voltage in kV')
    ctf.add_argument('--cs', type=float, default=None, help='Spherical aberration in mm')
    ctf.add_argument('--amp-contrast', type=float, default=None, help='Amplitude contrast')
    ctf.add_argument('--ctffind4', action='store_true',
                     help='Use CTFFIND4 prompts instead of the CTFFIND5 default')
    ctf.add_argument('--bin', type=int, default=1,
                     help='Binning factor for the left/right aligned stacks')
    ctf.add_argument('--exclude-negative', type=int, default=0, dest='exclude_negative',
                     help='Tilts to exclude from the negative end of the series')
    ctf.add_argument('--exclude-positive', type=int, default=0, dest='exclude_positive',
                     help='Tilts to exclude from the positive end of the series')

    tools = p.add_argument_group('tool paths (remembered in project.json)')
    tools.add_argument('--imod-dir', default=None, help='IMOD install dir (default: /opt/IMOD)')
    tools.add_argument('--ctffind-bin', default=None,
                       help='ctffind binary (default: CTFFIND5 unless --ctffind4)')
    tools.add_argument('--defocusgrad-bin', default=None,
                       help=f'defocusgrad script path (default: {_DEFAULT_DEFOCUSGRAD_BIN})')

    p.set_defaults(func=run)
    return p


def run(args):
    analysis_dir = Path(args.analysis) if args.analysis else get_latest_analysis_dir()
    if analysis_dir is None:
        print('ERROR: --analysis not given and no analyse run found in project.json.')
        sys.exit(1)
    analysis_dir = Path(analysis_dir)
    json_path = analysis_dir / 'alignment_data.json'
    if not json_path.exists():
        print(f'ERROR: alignment_data.json not found in {analysis_dir}')
        sys.exit(1)

    # aln_dir (where ts-XXX_Imod/ts-XXX_st.{xf,tlt} actually live) is
    # analyse's own --input, which is NOT the same directory as --analysis
    # above (analyse's --output) in general -- they only happen to nest
    # together when --output was pointed inside the AreTomo3 dir. Recover
    # it from analyse's own recorded args rather than assuming.
    if args.aln_dir:
        aln_dir = Path(args.aln_dir)
    else:
        proj = load_project()
        recorded_input = proj.get('analyse', {}).get('args', {}).get('input')
        if not recorded_input:
            print('ERROR: --aln-dir not given and project.json has no recorded '
                  'analyse.args.input to default from. Pass --aln-dir explicitly '
                  '(the directory analyse was run with --input against).')
            sys.exit(1)
        aln_dir = Path(recorded_input)
    if not aln_dir.is_dir():
        print(f'ERROR: --aln-dir {aln_dir} not found')
        sys.exit(1)

    cmd0_dir = Path(args.cmd0_dir) if args.cmd0_dir else get_cmd0_outdir()
    if cmd0_dir is None:
        print('ERROR: --cmd0-dir not given and no input_stacks.cmd0_outdir in project.json.')
        sys.exit(1)
    cmd0_dir = Path(cmd0_dir)

    with open(json_path) as fh:
        alignment_data = json.load(fh)
    alignment_data = {k: v for k, v in alignment_data.items() if not k.startswith('[')}

    # ── Tool paths ────────────────────────────────────────────────────────
    imod_dir = resolve_tool_path('imod', args.imod_dir) or _DEFAULT_IMOD_DIR
    newstack_bin = str(Path(imod_dir) / 'bin' / 'newstack')
    default_ctffind = _DEFAULT_CTFFIND4_BIN if args.ctffind4 else _DEFAULT_CTFFIND5_BIN
    ctffind_bin = resolve_tool_path('ctffind', args.ctffind_bin) or default_ctffind
    defocusgrad_bin = resolve_tool_path('defocusgrad', args.defocusgrad_bin) or _DEFAULT_DEFOCUSGRAD_BIN

    for label, path in (('newstack (IMOD)', newstack_bin), ('ctffind', ctffind_bin),
                        ('defocusgrad', defocusgrad_bin)):
        if not Path(path).is_file():
            print(f'ERROR: {label} not found: {path}')
            print(f'       Set the appropriate --imod-dir/--ctffind-bin/--defocusgrad-bin flag.')
            sys.exit(1)

    # ── CTF params: explicit > run_aretomo3_params > defocusgrad's own defaults ──
    run_params = get_run_params() or {}
    kv  = args.kv  if args.kv  is not None else run_params.get('kv', 300.0)
    cs  = args.cs  if args.cs  is not None else run_params.get('cs', 2.7)
    ac  = args.amp_contrast if args.amp_contrast is not None else run_params.get('amp_contrast', 0.07)
    args.kv, args.cs, args.amp_contrast = kv, cs, ac

    # ── TS selection ─────────────────────────────────────────────────────
    if args.select_ts:
        missing = [t for t in args.select_ts if t not in alignment_data]
        if missing:
            print(f'ERROR: --select-ts names not found in alignment_data.json: {missing}')
            sys.exit(1)
        scores = _score_tilt_series(alignment_data)
        selected = [(t, scores.get(t, {})) for t in args.select_ts]
    else:
        selected = _select_ts(alignment_data, args.n_ts)
        if not selected:
            print('ERROR: no tilt series had enough per-frame CTF data to score for selection.')
            sys.exit(1)

    print(f'Selected {len(selected)} TS (of {len(alignment_data)}):')
    for ts_name, s in selected:
        print(f'  {ts_name}: coverage={s.get("coverage_deg")}°  '
              f'high-tilt CTF res={s.get("high_tilt_ctf_res_A")}Å  '
              f'overall CTF res={s.get("overall_ctf_res_A")}Å')

    out_dir = check_output_dir(Path(args.output).resolve(), clean=args.clean, dry_run=args.dry_run)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve + validate inputs for every selected TS up front (cheap,
    #    file-existence only) so a missing file fails fast without wasting
    #    a parallel worker slot on a job that can't run anyway ────────────
    per_ts = {}
    selection_scores = {name: s for name, s in selected}
    jobs = []
    for ts_name, _ in selected:
        st_path  = cmd0_dir / f'{ts_name}.mrc'
        xf_path  = aln_dir / f'{ts_name}_Imod' / f'{ts_name}_st.xf'
        tlt_path = aln_dir / f'{ts_name}_Imod' / f'{ts_name}_st.tlt'
        missing = [str(p) for p in (st_path, xf_path, tlt_path) if not p.exists()]
        if missing:
            print(f'\n[{ts_name}] ERROR: missing input file(s): {missing}')
            print(f'  (the _Imod/*.xf/*.tlt files need "run-aretomo3 --out-imod 1", the default)')
            per_ts[ts_name] = {'handedness': None, 'error': 'missing input files'}
            continue
        jobs.append((ts_name, st_path, xf_path, tlt_path))

    if args.dry_run:
        for ts_name, st_path, xf_path, tlt_path in jobs:
            _run_one_ts(ts_name, st_path, xf_path, tlt_path, out_dir, args,
                       defocusgrad_bin, newstack_bin, ctffind_bin)
        print('\n[dry-run: no defocusgrad invocations were made]')
        return

    # ── Run (in parallel by default -- each job is its own subprocess, so
    #    threads are just for orchestration/waiting, not real GIL contention) ──
    max_workers = args.jobs if args.jobs > 0 else max(1, len(jobs))
    if jobs:
        print(f'\nRunning {len(jobs)} TS with up to {max_workers} in parallel...')
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_ts, ts_name, st_path, xf_path, tlt_path, out_dir, args,
                       defocusgrad_bin, newstack_bin, ctffind_bin): ts_name
            for ts_name, st_path, xf_path, tlt_path in jobs
        }
        for future in as_completed(futures):
            ts_name = futures[future]
            r = future.result()
            per_ts[ts_name] = r
            h = r.get('handedness')
            status = f'{h:+d}' if h in (-1, 1) else ('inconclusive' if h == 0 else r.get('error', '?'))
            ctf_l, ctf_r = r.get('ctf_left'), r.get('ctf_right')
            ctf_note = (f'  (CTFFIND res: left={ctf_l["res_mean_A"]:.1f}Å right={ctf_r["res_mean_A"]:.1f}Å)'
                       if ctf_l and ctf_r else '')
            print(f'[{ts_name}] done -> {status}{ctf_note}')

    consensus = _consensus(per_ts)
    print(f'\nConsensus defocus handedness: {consensus}')

    html_path = out_dir / 'index.html'
    _make_html(per_ts, selection_scores, consensus, html_path)

    update_section('defocusgrad', {
        'command':   ' '.join(sys.argv),
        'args':      args_to_dict(args),
        'output_dir': str(out_dir),
        'consensus': consensus,
        'per_ts': {ts: {**r, 'selection': selection_scores.get(ts, {})} for ts, r in per_ts.items()},
    }, backup_dir=out_dir)

    record_analysis_run('defocusgrad', str(out_dir))
    landing_path = write_landing_page(Path.cwd())

    print(f'\nOutput')
    print(f'  Report       : {html_path}')
    print(f'  Report index : {landing_path}')
