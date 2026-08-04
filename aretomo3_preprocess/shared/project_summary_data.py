"""
project_summary_data.py — project.json reading/computation shared by
`project-summary` (terminal) and `shared/landing_page.py` (HTML "first page"
panel on analysis_start.html).

`gather(proj)` reads an already-loaded project.json dict and returns a
plain, JSON-serialisable-ish dict -- no printing, no colour, no HTML. Both
renderers call it so the CLI and the HTML summary can never disagree with
each other about what a value is; each renderer only decides how to display
the same numbers.

`render_html(proj)` renders that data as an HTML fragment (tables grouped
into blocks, no histograms -- meant to be a compact glance, not a full
report). Lives here rather than in commands/project_summary.py so that
shared/landing_page.py can import it without commands/ importing back into
shared/ (this codebase has a history of exactly that kind of layering
violation -- see CLAUDE.md).
"""

import html
from pathlib import Path
from collections import Counter

from aretomo3_preprocess.shared.project_json import SCHEMA_VERSION


def _mode(vals):
    clean = [v for v in vals if v is not None]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _mdoc_representative_ts(per_ts):
    """One TS's mdoc frame data to use for tilt-range display -- these are
    per-acquisition-scheme values, not worth reporting per-TS, so a single
    representative sample (first by name) is shown rather than an aggregate
    across every TS."""
    if not per_ts:
        return None, None
    ts_key = sorted(per_ts)[0]
    return ts_key, per_ts[ts_key]


def gather(proj: dict) -> dict:
    """
    Read every value project-summary reports from an already-loaded
    project.json dict and return it as a plain dict (no printing, no
    colour, no HTML).
    """
    d = {}
    d['schema_version'] = proj.get('project', {}).get('schema_version', 0)
    d['last_updated']   = proj.get('project', {}).get('last_updated', '?')

    mdoc   = proj.get('mdoc_data', {})
    per_ts = mdoc.get('per_ts', {})
    rename = proj.get('rename_ts', {})
    lookup = rename.get('lookup', {})
    d['n_mdoc']    = len(per_ts)
    d['n_renamed'] = len(lookup)
    d['renaming_example'] = None
    if lookup:
        ts_mdoc_name, orig_path = sorted(lookup.items())[0]
        d['renaming_example'] = (Path(ts_mdoc_name).stem, Path(orig_path).stem)

    angpix_vals = [v.get('angpix') for v in per_ts.values() if v.get('angpix') is not None]
    voltages = [v.get('acquisition', {}).get('voltage') for v in per_ts.values()
               if v.get('acquisition', {}).get('voltage') is not None]
    d['angpix_mdoc']   = _mode(angpix_vals)
    d['voltage_mode']  = _mode(voltages)
    d['calibrated']    = proj.get('calibrated_apix', {})

    widths  = [v.get('acquisition', {}).get('width')  for v in per_ts.values()]
    heights = [v.get('acquisition', {}).get('height') for v in per_ts.values()]
    d['image_dims'] = (_mode(widths), _mode(heights))

    ts_key, ts_sample = _mdoc_representative_ts(per_ts)
    d['ts_key'] = ts_key
    d['frames_per_tilt'] = None
    d['tilt_range'] = None
    n_sub_mode = n_t = None
    if ts_sample is not None:
        frames = ts_sample.get('frames', {})
        n_sub_mode = _mode([f.get('num_subframes') for f in frames.values()])
        d['frames_per_tilt'] = n_sub_mode

        tilts = [f.get('tilt_angle') for f in frames.values() if f.get('tilt_angle') is not None]
        n_t = len(tilts)
        if tilts:
            tilts_sorted = sorted(tilts)
            step = (tilts_sorted[-1] - tilts_sorted[0]) / (n_t - 1) if n_t > 1 else 0.0
            d['tilt_range'] = (tilts_sorted[0], tilts_sorted[-1], n_t, step)

    # Dose: deliberately NOT from mdoc's ExposureDose (unreliable --
    # planning-time value, not what was actually used) -- the real per-frame
    # dose is whatever was passed to `run-aretomo3 --fm-dose` for the cmd=0
    # (motion correction) run. project.json's `run_aretomo3` section is
    # last-write-wins across *any* cmd though, so this is only trustworthy
    # when the most recent run-aretomo3 invocation was itself cmd=0 --
    # checked explicitly rather than assumed.
    run3 = proj.get('run_aretomo3', {})
    fm_dose  = run3.get('args', {}).get('fm_dose')
    run3_cmd = run3.get('args', {}).get('cmd')
    if fm_dose is not None and run3_cmd == 0 and n_sub_mode and n_t:
        per_tilt = fm_dose * n_sub_mode
        d['dose'] = {'status': 'ok', 'per_frame': fm_dose, 'per_tilt': per_tilt,
                     'total': per_tilt * n_t, 'timestamp': run3.get('timestamp', '?')}
    elif fm_dose is not None:
        d['dose'] = {'status': 'stale_cmd', 'per_frame': fm_dose, 'cmd': run3_cmd}
    else:
        d['dose'] = {'status': 'missing'}

    d['ctf'] = proj.get('run_aretomo3_params') or None
    d['gain_check'] = proj.get('gain_check') or None

    stacks = proj.get('input_stacks', {})
    d['paths'] = {
        'cmd0_outdir': stacks.get('cmd0_outdir'),
        'n_stacks':    stacks.get('n_stacks'),
        'tlt_dir':     stacks.get('tlt_dir'),
        'tool_paths':  proj.get('tool_paths', {}),
    }

    d['handedness'] = proj.get('handedness')
    d['defocusgrad'] = proj.get('defocusgrad')
    d['ctf_handedness'] = proj.get('ctf_handedness')
    lamellae = proj.get('lamella_assignments', {})
    d['lamellae'] = None
    if lamellae:
        d['lamellae'] = (len(set(lamellae.get('positions', {}).values())), lamellae.get('n_ts', '?'))

    d['analysis_runs'] = proj.get('analysis_runs', [])

    analyse_sec = proj.get('analyse', {})
    d['analyse_sec']  = analyse_sec
    d['qc_summary']   = analyse_sec.get('qc_summary')
    d['per_ts_qc']    = analyse_sec.get('per_ts_qc')

    return d


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering (embedded in shared/landing_page.py's analysis_start.html)
# ─────────────────────────────────────────────────────────────────────────────

def _e(x) -> str:
    return html.escape(str(x))


def _html_row(label, value_html, note=None):
    note_html = f'<div class="ps-note">{_e(note)}</div>' if note else ''
    return f'<tr><td class="ps-label">{_e(label)}</td><td class="ps-value">{value_html}{note_html}</td></tr>'


def render_html(proj: dict) -> str:
    """
    Render the same information project-summary prints to the terminal
    (minus histograms -- this is meant to be a compact glance, not a full
    report) as an HTML fragment: a series of `<table class="ps-table">`
    blocks, one per section, meant to be dropped into
    shared/landing_page.py's analysis_start.html.
    """
    d = gather(proj)
    blocks = []

    # ── mdoc & renaming ───────────────────────────────────────────────────
    rows = [_html_row('mdoc files cached', _e(d['n_mdoc'])),
            _html_row('ts-XXX renamed', _e(d['n_renamed']))]
    if d['renaming_example']:
        ts_stem, orig_stem = d['renaming_example']
        rows.append(_html_row('renaming example', f'{_e(ts_stem)} &rarr; {_e(orig_stem)}'))

    cal = d['calibrated']
    cal_value = cal.get('value')
    angpix_mdoc = d['angpix_mdoc']
    if cal_value is not None:
        bracket = f' <span class="ps-dim">({_e(angpix_mdoc)} from mdoc)</span>' if (
            angpix_mdoc is not None and abs(angpix_mdoc - cal_value) > 1e-6) else ''
        rows.append(_html_row('pixel size', f'<span class="ps-good">{_e(cal_value)} &Aring;/px</span>{bracket}',
                              note=f'calibrated via {cal.get("source", "?")}'))
    elif angpix_mdoc is not None:
        rows.append(_html_row('pixel size', f'<span class="ps-warn">{_e(angpix_mdoc)} &Aring;/px (raw mdoc, not yet calibrated)</span>'))
    else:
        rows.append(_html_row('pixel size', 'n/a'))

    w_mode, h_mode = d['image_dims']
    if d['voltage_mode']:
        rows.append(_html_row('voltage', f'{d["voltage_mode"]:.0f} kV'))
    if w_mode and h_mode:
        rows.append(_html_row('image dimensions', f'{_e(w_mode)} &times; {_e(h_mode)} px'))
    if d['frames_per_tilt'] is not None:
        rows.append(_html_row('frames per tilt', _e(d['frames_per_tilt'])))
    if d['tilt_range']:
        lo, hi, n_t, step = d['tilt_range']
        rows.append(_html_row(f'tilt range (from {_e(d["ts_key"])})',
                              f'{lo:+.1f}&deg; &rarr; {hi:+.1f}&deg;  ({n_t} tilts, ~{step:.2f}&deg; step)'))

    dose = d['dose']
    if dose['status'] == 'ok':
        rows.append(_html_row(f'dose (--fm-dose, cmd=0)',
                              f'{dose["per_frame"]:.3f} e&#8315;/&Aring;&sup2; per frame &middot; '
                              f'{dose["per_tilt"]:.2f} per tilt &middot; ~{dose["total"]:.0f} total',
                              note=dose['timestamp']))
    elif dose['status'] == 'stale_cmd':
        rows.append(_html_row('dose', f'<span class="ps-warn">--fm-dose={dose["per_frame"]} e&#8315;/&Aring;&sup2; '
                              f'but from cmd={dose["cmd"]}, not cmd=0 -- may be stale</span>'))
    blocks.append(('mdoc & renaming', rows))

    # ── CTF / optics ──────────────────────────────────────────────────────
    if d['ctf']:
        rp = d['ctf']
        rows = [
            _html_row('voltage', f'{_e(rp.get("kv", "n/a"))} kV'),
            _html_row('spherical aberration (Cs)', f'{_e(rp.get("cs", "n/a"))} mm'),
            _html_row('amplitude contrast', _e(rp.get('amp_contrast', 'n/a'))),
            _html_row('pixel size used', f'{_e(rp.get("apix", "n/a"))} &Aring;/px', note=f'cmd {rp.get("cmd", "?")}'),
        ]
        blocks.append(('CTF / optics params', rows))

    # ── Gain transform ────────────────────────────────────────────────────
    if d['gain_check']:
        gc = d['gain_check']
        rows = [
            _html_row('best transform', f'<span class="ps-good">{_e(gc.get("best_transform", "n/a"))}</span>'),
            _html_row('AreTomo3 flags', f'-RotGain {_e(gc.get("aretomo3_rot_gain", "?"))} '
                      f'-FlipGain {_e(gc.get("aretomo3_flip_gain", "?"))}'),
        ]
        blocks.append(('Gain transform', rows))

    # ── Paths ─────────────────────────────────────────────────────────────
    paths = d['paths']
    rows = [
        _html_row('cmd0 output dir', _e(paths['cmd0_outdir'] or 'n/a')),
        _html_row('mrc stacks registered', _e(paths['n_stacks'] if paths['n_stacks'] is not None else 'n/a')),
        _html_row('TLT dir', _e(paths['tlt_dir'] or 'n/a'),
                  note='only refreshed by "run-aretomo3 --cmd 0"' if paths['tlt_dir'] else None),
    ]
    for tool in ('aretomo3', 'pytom', 'imod', 'gapstop'):
        val = paths['tool_paths'].get(tool)
        rows.append(_html_row(f'{tool} path', _e(val) if val else '<span class="ps-dim">not recorded</span>'))
    blocks.append(('Paths', rows))

    # ── Handedness / lamellae ────────────────────────────────────────────
    handedness = d['handedness']
    rows = []
    if handedness:
        mirror = handedness.get('mirror')
        cls = 'ps-warn' if mirror else 'ps-good'
        txt = 'YES &mdash; use --mirror' if mirror else 'no mirroring needed'
        rows.append(_html_row('physical (mirror) handedness', f'<span class="{cls}">{txt}</span>',
                              note=f'{handedness.get("particle", "?")}, {len(handedness.get("per_ts", {}))} TS'))
    else:
        rows.append(_html_row('physical (mirror) handedness',
                              '<span class="ps-dim">not determined yet (pytom-ribo-auto --check-handedness)</span>'))
    dg = d['defocusgrad']
    if dg:
        consensus = dg.get('consensus', '?')
        cls = 'ps-warn' if 'inconsist' in str(consensus) or 'inconclus' in str(consensus) else 'ps-good'
        n_ts = len(dg.get('per_ts', {}))
        rows.append(_html_row('defocus handedness', f'<span class="{cls}">{_e(consensus)}</span>',
                              note=f'defocusgrad, {n_ts} TS'))
    else:
        rows.append(_html_row('defocus handedness',
                              '<span class="ps-dim">not checked yet (defocusgrad)</span>'))
    ch = d['ctf_handedness']
    if ch:
        consensus = ch.get('consensus', '?')
        cls = 'ps-warn' if 'inconsist' in str(consensus) or 'inconclus' in str(consensus) else 'ps-good'
        n_ts = len(ch.get('per_ts', {}))
        rows.append(_html_row('defocus handedness (ctfplotter)', f'<span class="{cls}">{_e(consensus)}</span>',
                              note=f'ctf-handedness, {n_ts} TS'))
    else:
        rows.append(_html_row('defocus handedness (ctfplotter)',
                              '<span class="ps-dim">not checked yet (ctf-handedness)</span>'))
    if d['lamellae']:
        n_lam, n_ts_assigned = d['lamellae']
        rows.append(_html_row('lamellae', f'{n_lam}  ({_e(n_ts_assigned)} TS assigned)'))
    blocks.append(('Handedness & lamellae', rows))

    # ── QC distributions (ranges only, no histograms) ─────────────────────
    qc_summary = d['qc_summary']
    analyse_sec = d['analyse_sec']
    if qc_summary:
        n_ts = analyse_sec.get('n_tilt_series', len(d['per_ts_qc'] or []))
        rows = []
        for key, label, unit in (('tilt_axis_deg', 'median tilt axis (ROT)', '&deg;'),
                                 ('mean_overlap', 'mean frame overlap', '%'),
                                 ('thickness_nm', 'thickness', 'nm'),
                                 ('avg_defocus_um', 'defocus', '&micro;m')):
            r = qc_summary.get(key)
            if r:
                rows.append(_html_row(label, f'{r["lo"]:.2f} &ndash; {r["hi"]:.2f} {unit}  '
                                      f'(median {r["median"]:.2f})'))
        if rows:
            blocks.append((f'QC distributions &mdash; {n_ts} TS, 2nd&ndash;98th pctile trimmed', rows))

    table_html = ''.join(f"""
      <div class="ps-block">
        <div class="ps-block-title">{title}</div>
        <table class="ps-table">{''.join(rows)}</table>
      </div>""" for title, rows in blocks)

    schema_note = ''
    if d['schema_version'] and d['schema_version'] < SCHEMA_VERSION:
        schema_note = (f'<div class="ps-schema-warn">project.json schema v{d["schema_version"]} '
                       f'(expected v{SCHEMA_VERSION}) -- some fields may be missing until the '
                       f'relevant command is re-run.</div>')

    return f"""
  <details id="project-summary-panel" open>
    <summary>Project summary <span class="ps-updated">last updated {_e(d["last_updated"])}</span></summary>
    {schema_note}
    <div class="ps-grid">{table_html}</div>
  </details>"""
