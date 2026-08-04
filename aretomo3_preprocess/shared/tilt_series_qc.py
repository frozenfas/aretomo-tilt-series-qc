"""
tilt_series_qc.py — shared TS auto-selection / section-filtering helpers for
the defocus-handedness QC commands (defocusgrad, ctf_handedness).

Both commands need to pick the same handful of "good" tilt series (wide
angular coverage, good CTF fit quality, especially at high tilts) so their
verdicts are directly comparable rather than each judging a different
subset of the dataset -- see CLAUDE.md's frame cross-referencing convention
for why sec/frame_b (not tilt-angle matching) is used for section indexing.
"""

import os
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TS auto-selection
# ─────────────────────────────────────────────────────────────────────────────

def score_tilt_series(alignment_data: dict) -> dict:
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


def select_ts(alignment_data: dict, n_ts: int, coverage_pctile: float = 75.0) -> list:
    """
    Pick n_ts TS: keep those with coverage >= coverage_pctile among all TS
    (the "high tilt coverage" qualifying filter), then rank survivors by
    high-tilt CTF resolution ascending (best first), overall CTF resolution
    as tie-breaker. Returns a list of (ts_name, score_dict), best first.
    """
    scores = score_tilt_series(alignment_data)
    if not scores:
        return []
    coverages = [s['coverage_deg'] for s in scores.values()]
    coverage_thresh = float(np.percentile(coverages, coverage_pctile))
    survivors = [(name, s) for name, s in scores.items() if s['coverage_deg'] >= coverage_thresh]
    survivors.sort(key=lambda t: (t[1]['high_tilt_ctf_res_A'], t[1]['overall_ctf_res_A']))
    return survivors[:n_ts]


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess environment for IMOD / CTFFIND / ctfplotter tools
# ─────────────────────────────────────────────────────────────────────────────

def imod_env(newstack_bin: str, ctffind_bin: str = None) -> dict:
    """
    Env for subprocesses that shell out to `newstack` (and optionally
    `ctffind`) by bare name -- both binaries' directories need to be on
    PATH, and IMOD_DIR needs to be set for newstack's own wrapper script to
    find its "realbin" -- same pattern as pytom_ribo_auto.py's
    _resample_volume(). ctffind_bin is optional: ctfplotter-based callers
    don't shell out to ctffind at all.

    MPLBACKEND=Agg is also required for any caller that plots: the
    defocusgrad script calls plt.show() unconditionally after plt.savefig()
    with no backend override of its own -- without this, matplotlib picks
    an interactive backend and plt.show() hangs forever waiting for a GUI
    that can never appear in a subprocess (confirmed: it genuinely hung,
    not just slow).
    """
    imod_dir = str(Path(newstack_bin).resolve().parent.parent)  # .../bin/newstack -> ...
    env = dict(os.environ)
    env['IMOD_DIR'] = os.environ.get('IMOD_DIR', imod_dir)
    path_dirs = [str(Path(newstack_bin).parent)]
    if ctffind_bin:
        path_dirs.append(str(Path(ctffind_bin).parent))
    env['PATH'] = f"{':'.join(path_dirs)}:{env.get('PATH', '')}"
    env['MPLBACKEND'] = 'Agg'
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Excluding dark / low-overlap sections
# ─────────────────────────────────────────────────────────────────────────────
# DefocusGrad's own internal newstack calls have no -secs option -- they
# always use every section of whatever --st stack they're given. To exclude
# specific dark/low-overlap sections (which can be anywhere in the tilt
# series, not just at the ends -- --exclude_negative/--exclude_positive only
# trim from the two ends) we build a smaller raw stack ourselves first via a
# separate `newstack -secs` call, plus a matching trimmed .xf/.tlt, and hand
# *that* to defocusgrad instead of the original full stack.

def sec_to_idx(sec_numbers, n_secs):
    """1-indexed AreTomo3 SEC numbers -> 0-indexed newstack section indices
    -- same exact convention as trim_ts.py's sections_from_sec_numbers()
    (sec/frame_b are AreTomo3's own tilt-sorted-stack section numbers, a
    direct index lookup, not a tilt-angle match -- see CLAUDE.md)."""
    return sorted({s - 1 for s in sec_numbers if 0 <= s - 1 < n_secs})


def good_sections(ts_data: dict, min_overlap, exclude_dark: bool) -> dict:
    """
    0-indexed sections to keep for one TS's alignment_data.json entry,
    after excluding dark frames (if exclude_dark) and/or frames with
    overlap_pct below min_overlap (if given). total_frames = len(frames) +
    len(dark_frames) always (verified against real data).
    """
    frames = ts_data.get('frames', [])
    n_total = ts_data.get('total_frames') or (len(frames) + len(ts_data.get('dark_frames', [])))
    all_idx = set(range(n_total))

    dark_idx = set()
    if exclude_dark:
        dark_secnums = [df['frame_b'] for df in ts_data.get('dark_frames', [])]
        dark_idx = set(sec_to_idx(dark_secnums, n_total))

    lowov_idx = set()
    if min_overlap is not None:
        lowov_secnums = [f['sec'] for f in frames
                         if f.get('overlap_pct') is not None and f['overlap_pct'] < min_overlap]
        lowov_idx = set(sec_to_idx(lowov_secnums, n_total)) - dark_idx

    return {
        'keep_idx':  sorted(all_idx - dark_idx - lowov_idx),
        'n_total':   n_total,
        'n_dark':    len(dark_idx),
        'n_low_overlap': len(lowov_idx),
    }
