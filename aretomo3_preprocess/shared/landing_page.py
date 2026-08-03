"""
landing_page.py — project-level HTML index linking to every run's report.

write_landing_page(project_dir) reads the analysis_runs history recorded by
shared.project_state.record_analysis_run() and renders analysis_start.html
into project_dir (the same directory as aretomo3_project.json). One tab per
run label, each tab listing whichever of gain-check/aretomo-analyse/pytom
reports that run actually has -- as plain links to the existing
self-contained report files (report.html / index.html / pytom_*_qc*.html),
not iframes, so each individual report stays independently portable.

Deliberately cheap to regenerate: callers (check_gain_transform.py,
analyse.py, pytom_match.py, gapstop_match.py) call this at the end of every
run, right after record_analysis_run(), so it never goes stale and needs no
separate command -- same pattern project_json.update_section() already
uses for auto-fill state elsewhere in this codebase.
"""

import html
from pathlib import Path

from aretomo3_preprocess.shared.project_json import load as load_project, PROJECT_FILENAME
from aretomo3_preprocess.shared.project_state import get_analysis_runs
from aretomo3_preprocess.shared.project_summary_data import render_html as render_project_summary_html

_KIND_LABEL = {
    'gain_check':      ('Gain check',         '🎯'),
    'aretomo_analyse':  ('AreTomo3 analysis',  '📊'),
    'pytom_match':      ('Pytom picking',      '🧬'),
    'gapstop_match':    ('Gapstop picking',    '🧬'),
}

# Report file(s) each kind links to, relative to that run's own output_dir --
# (link_text, filename), skipped if the file doesn't actually exist on disk.
_KIND_FILES = {
    'gain_check':     [('Report', 'report.html')],
    'aretomo_analyse': [('Report', 'index.html')],
    'pytom_match':     [('Match QC (raw score map)', 'pytom_match_qc.html'),
                        ('Picks (interactive)',      'pytom_extract_qc_dev.html')],
    'gapstop_match':   [('Match QC (raw score map)', 'gapstop_match_qc.html'),
                        ('Picks (interactive)',      'gapstop_extract_qc_dev.html')],
}


def write_landing_page(project_dir) -> Path:
    """Render analysis_start.html into project_dir. Returns the written path."""
    project_dir = Path(project_dir)
    runs = get_analysis_runs()

    # Group by label (a "run"), preserving first-seen order; each label can
    # carry more than one kind (e.g. a run that has both gain_check and an
    # aretomo_analyse entry pointing at output dirs under the same label).
    by_label = {}
    for r in runs:
        by_label.setdefault(r['label'], []).append(r)

    tab_buttons, tab_sections = [], []
    for i, (label, entries) in enumerate(by_label.items()):
        tab_id = f'run{i}'
        active = ' active' if i == 0 else ''
        # Most-recent timestamp among this tab's entries -- shown on the
        # button itself so runs sharing a kind (e.g. several cmd1 `analyse`
        # attempts) are distinguishable without opening every tab.
        latest_ts = max((e.get('timestamp', '') for e in entries), default='')
        kinds_here = sorted({e['kind'] for e in entries})
        icons = ''.join(_KIND_LABEL.get(k, (k, '📄'))[1] for k in kinds_here)
        tab_buttons.append(
            f'<button class="tab-btn{active}" data-tab="{tab_id}" '
            f'onclick="switchTab(\'{tab_id}\')">'
            f'<span class="tab-btn-label">{icons} {html.escape(label)}</span>'
            f'<span class="tab-btn-time">{html.escape(latest_ts)}</span>'
            f'</button>'
        )

        cards = []
        for entry in sorted(entries, key=lambda e: e['kind']):
            kind = entry['kind']
            kind_label, icon = _KIND_LABEL.get(kind, (kind, '📄'))
            out_dir = (project_dir / entry['output_dir']).resolve()
            # Relative to project_dir when possible -- callers pass
            # str(Path(args.output).resolve()) (always absolute), and an
            # absolute href only resolves correctly under file://, not
            # http:// (this codebase's ratings-CSV auto-reload already
            # supports serving reports over a local HTTP server). Falls
            # back to the absolute path if output_dir genuinely isn't
            # under project_dir (e.g. a different drive/mount).
            try:
                out_dir_rel = out_dir.relative_to(project_dir.resolve())
            except ValueError:
                out_dir_rel = out_dir
            links = []
            for link_text, fname in _KIND_FILES.get(kind, []):
                fpath = out_dir / fname
                if fpath.exists():
                    href = f'{out_dir_rel.as_posix()}/{fname}'
                    links.append(f'<a href="{html.escape(href)}">{html.escape(link_text)}</a>')
            if not links:
                links.append('<span class="missing">no report file found</span>')
            cards.append(f"""
      <div class="run-card">
        <div class="run-card-title">{icon} {html.escape(kind_label)}</div>
        <div class="run-card-path">{html.escape(entry['output_dir'])}</div>
        <div class="run-card-links">{' &middot; '.join(links)}</div>
        <div class="run-card-time">{html.escape(entry.get('timestamp', ''))}</div>
      </div>""")

        tab_sections.append(f"""
  <div id="tab-{tab_id}" class="tab-section" style="{'' if i == 0 else 'display:none'}">
    {''.join(cards)}
  </div>""")

    body = ('<p class="empty">No analysis runs recorded yet in this project. '
            'Run analyse / check-gain-transform / pytom-match / pytom-ribo-auto '
            'to populate this page.</p>') if not runs else (
        f'<div id="tab-bar">{"".join(tab_buttons)}</div>{"".join(tab_sections)}'
    )

    proj = load_project(project_dir / PROJECT_FILENAME)
    summary_html = render_project_summary_html(proj) if proj else ''

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AreTomo3-Preprocess — {html.escape(project_dir.name)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', sans-serif; background: #ffffff; color: #263238;
      display: flex; flex-direction: column; align-items: center;
      padding: 32px 16px; min-height: 100vh;
    }}
    h1 {{ margin-bottom: 2px; font-size: 1.4em; color: #0d47a1; letter-spacing: 0.03em; }}
    #project-path {{
      margin-bottom: 24px; font-size: 0.8em; color: #78909c;
      font-family: monospace; word-break: break-all;
    }}
    #tab-bar {{
      display: flex; gap: 6px; margin-bottom: 24px; flex-wrap: wrap;
      justify-content: center; width: 100%; max-width: 1000px;
    }}
    .tab-btn {{
      display: flex; flex-direction: column; align-items: center; gap: 2px;
      padding: 6px 20px; font-size: 0.92em; border: none; border-radius: 6px;
      background: #eceff1; color: #546e7a; cursor: pointer; transition: all 0.15s;
    }}
    .tab-btn.active {{ background: #1565c0; color: white; }}
    .tab-btn.active .tab-btn-time {{ color: #bbdefb; }}
    .tab-btn:hover:not(.active) {{ background: #cfd8dc; color: #263238; }}
    .tab-btn-label {{ font-weight: 500; }}
    .tab-btn-time {{ font-size: 0.78em; color: #90a4ae; font-family: monospace; }}
    .tab-section {{
      width: 100%; max-width: 1000px;
      display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .run-card {{
      background: #f5f7fa; border: 1px solid #e0e6ea; border-radius: 10px;
      padding: 18px 20px; transition: box-shadow 0.15s;
    }}
    .run-card:hover {{ box-shadow: 0 2px 10px rgba(13,71,161,0.12); }}
    .run-card-title {{ font-size: 1.05em; font-weight: 600; color: #263238; margin-bottom: 4px; }}
    .run-card-path {{
      font-family: monospace; font-size: 0.76em; color: #90a4ae;
      word-break: break-all; margin-bottom: 10px;
    }}
    .run-card-links {{ font-size: 0.92em; }}
    .run-card-links a {{ color: #1565c0; text-decoration: none; font-weight: 500; }}
    .run-card-links a:hover {{ text-decoration: underline; }}
    .run-card-links .missing {{ color: #b0bec5; font-style: italic; }}
    .run-card-time {{ font-size: 0.72em; color: #b0bec5; margin-top: 10px; }}
    .empty {{ color: #78909c; margin-top: 40px; }}

    #project-summary-panel {{
      width: 100%; max-width: 1000px; margin-bottom: 24px;
      background: #f5f7fa; border: 1px solid #e0e6ea; border-radius: 10px;
      padding: 4px 20px 18px;
    }}
    #project-summary-panel summary {{
      cursor: pointer; padding: 14px 0; font-size: 1.05em; font-weight: 600;
      color: #263238; list-style: none;
    }}
    #project-summary-panel summary::-webkit-details-marker {{ display: none; }}
    #project-summary-panel summary::before {{ content: '▸ '; color: #1565c0; }}
    #project-summary-panel[open] summary::before {{ content: '▾ '; }}
    .ps-updated {{ font-weight: normal; font-size: 0.78em; color: #90a4ae; margin-left: 8px; }}
    .ps-schema-warn {{
      background: #fff8e1; color: #8d6e00; border: 1px solid #ffe082;
      border-radius: 6px; padding: 8px 12px; font-size: 0.82em; margin-bottom: 14px;
    }}
    .ps-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}
    .ps-block {{ background: #ffffff; border: 1px solid #e0e6ea; border-radius: 8px; padding: 12px 16px; }}
    .ps-block-title {{ font-size: 0.82em; font-weight: 600; color: #1565c0; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
    .ps-table {{ width: 100%; border-collapse: collapse; font-size: 0.86em; }}
    .ps-table td {{ padding: 3px 0; vertical-align: top; }}
    .ps-label {{ color: #78909c; padding-right: 10px; white-space: nowrap; }}
    .ps-value {{ color: #263238; }}
    .ps-note {{ font-size: 0.85em; color: #90a4ae; }}
    .ps-good {{ color: #2e7d32; font-weight: 600; }}
    .ps-warn {{ color: #b26a00; font-weight: 600; }}
    .ps-dim {{ color: #b0bec5; }}
  </style>
</head>
<body>
  <h1>AreTomo3-Preprocess</h1>
  <div id="project-path">{html.escape(str(project_dir))}</div>
  {summary_html}
  {body}
  <script>
    function switchTab(id) {{
      document.querySelectorAll('.tab-section').forEach(s => {{
        s.style.display = (s.id === 'tab-' + id) ? '' : 'none';
      }});
      document.querySelectorAll('.tab-btn').forEach(b => {{
        b.classList.toggle('active', b.dataset.tab === id);
      }});
    }}
  </script>
</body>
</html>
"""

    out_path = project_dir / 'analysis_start.html'
    out_path.write_text(html_out)
    return out_path
