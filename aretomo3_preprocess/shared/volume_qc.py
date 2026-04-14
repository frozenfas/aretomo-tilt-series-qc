"""
volume_qc.py — shared helpers for before/after tomogram quality-control HTML reports.

Used by imod-mtffilter, topaz-denoise3d, and any future processing commands that
want to visualise the effect of a processing step as side-by-side central-slab
projections.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Central slab projection
# ─────────────────────────────────────────────────────────────────────────────

def central_slab_projection(mrc_path, slab_angst: float = 300.0) -> Optional[dict]:
    """
    Return a mean XY projection through the central Z slab of a tomogram.

    Parameters
    ----------
    mrc_path   : path-like
    slab_angst : total slab thickness in Å (default 300 Å)

    Returns
    -------
    dict with keys:
        img     — 2-D float32 numpy array (ny × nx)
        vox     — voxel size in Å
        slab_a  — actual slab thickness used in Å
        nx, ny, nz
    or None on failure (mrcfile not installed, file missing, etc.)
    """
    try:
        import numpy as np
        import mrcfile
        with mrcfile.mmap(str(mrc_path), mode='r', permissive=True) as mrc:
            vox = float(mrc.voxel_size.x) or 1.0
            nz, ny, nx = mrc.data.shape
            hp  = max(1, int(round((slab_angst / 2.0) / vox)))
            zc  = nz // 2
            zs  = max(0,  zc - hp)
            ze  = min(nz, zc + hp)
            slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)
        return {
            'img':    slab.mean(axis=0),
            'vox':    vox,
            'slab_a': (ze - zs) * vox,
            'nx': nx, 'ny': ny, 'nz': nz,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PNG encoding
# ─────────────────────────────────────────────────────────────────────────────

def projection_to_b64png(
    img,
    pct: tuple = (1, 99),
    max_px: int = 800,
    cmap: str = 'gray',
    colorbar: bool = False,
) -> str:
    """
    Encode a 2-D float array as a base64 PNG suitable for embedding in HTML.

    Parameters
    ----------
    img      : 2-D numpy array
    pct      : (lo, hi) percentiles for contrast normalisation
    max_px   : downsample so the longest axis is at most this many pixels
    cmap     : matplotlib colormap name (default 'gray')
    colorbar : if True, add a colorbar on the right side of the image
    """
    import numpy as np
    from matplotlib.figure import Figure

    p_lo, p_hi = np.percentile(img, pct)
    span = float(p_hi - p_lo) or 1.0
    img_n = np.clip((img - p_lo) / span, 0, 1)

    # Downsample for HTML embedding if necessary
    h, w = img_n.shape
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        from PIL import Image as _PIL
        img_n = np.array(
            _PIL.fromarray((img_n * 255).astype('uint8')).resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                _PIL.LANCZOS,
            )
        ).astype(np.float32) / 255.0
        h, w = img_n.shape

    dpi  = 100
    figw = max(3.0, w / dpi + (0.6 if colorbar else 0))
    figh = max(2.0, h / dpi)
    fig  = Figure(figsize=(figw, figh), dpi=dpi)

    if colorbar:
        # Leave room for colorbar on the right
        img_frac = (w / dpi) / figw
        ax  = fig.add_axes([0, 0, img_frac - 0.02, 1])
        cax = fig.add_axes([img_frac + 0.01, 0.05, 0.06, 0.9])
        im  = ax.imshow(img_n, cmap=cmap, aspect='equal', interpolation='bilinear',
                        origin='lower', vmin=0, vmax=1)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels([f'{p_lo:.3g}', f'{(p_lo+p_hi)/2:.3g}', f'{p_hi:.3g}'])
        cb.ax.tick_params(labelsize=7, colors='white')
        cb.outline.set_edgecolor('white')
        fig.patch.set_facecolor('#111111')
    else:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_n, cmap=cmap, aspect='equal', interpolation='bilinear',
                  origin='lower', vmin=0, vmax=1)

    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation mask overlay
# ─────────────────────────────────────────────────────────────────────────────

def slab_with_mask_b64(
    tomo_path,
    mask_path,
    slab_angst: float = 300.0,
    color: tuple = (0.4, 0.6, 1.0),   # RGB 0–1, light blue
    alpha: float = 0.45,
    pct: tuple = (1, 99),
) -> Optional[str]:
    """
    Render the central Z slab of a tomogram with a segmentation mask overlaid
    as a semi-transparent coloured layer.

    Parameters
    ----------
    tomo_path  : path-like — tomogram MRC
    mask_path  : path-like — segmentation MRC (binary or labelled)
    slab_angst : total slab thickness in Å (default 300 Å)
    color      : RGB tuple 0–1 for the overlay colour (default light blue)
    alpha      : overlay opacity 0–1 (default 0.45)
    pct        : (lo, hi) percentiles for tomogram contrast

    Returns
    -------
    base64-encoded PNG str, or None on failure
    """
    try:
        import numpy as np
        import mrcfile
        from matplotlib.figure import Figure
    except ImportError:
        return None

    def _load_slab(path, zs, ze):
        with mrcfile.mmap(str(path), mode='r', permissive=True) as mrc:
            slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)
        return slab

    try:
        # ── Tomogram ─────────────────────────────────────────────────────────
        with mrcfile.mmap(str(tomo_path), mode='r', permissive=True) as mrc:
            vox = float(mrc.voxel_size.x) or 1.0
            nz, ny, nx = mrc.data.shape
            hp  = max(1, int(round((slab_angst / 2.0) / vox)))
            zc  = nz // 2
            zs  = max(0,  zc - hp)
            ze  = min(nz, zc + hp)
            tomo_slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)

        tomo_img = tomo_slab.mean(axis=0)   # (ny, nx)

        # ── Mask ─────────────────────────────────────────────────────────────
        # Use max-projection so any membrane voxel in the slab is shown
        with mrcfile.mmap(str(mask_path), mode='r', permissive=True) as mrc:
            mnz, mny, mnx = mrc.data.shape
            mzs = max(0,    int(zs * mnz / nz))
            mze = min(mnz,  int(ze * mnz / nz))
            mze = max(mze, mzs + 1)
            mask_slab = np.asarray(mrc.data[mzs:mze], dtype=np.float32)

        mask_img = mask_slab.max(axis=0)    # (mny, mnx)

        # Resize mask to match tomo if pixel sizes differ
        if mask_img.shape != tomo_img.shape:
            from PIL import Image as _PIL
            mask_img = np.array(
                _PIL.fromarray(mask_img).resize((nx, ny), _PIL.NEAREST)
            )

        # ── Render ───────────────────────────────────────────────────────────
        p_lo, p_hi = np.percentile(tomo_img, pct)
        tomo_n = np.clip((tomo_img - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)

        # Convert grayscale tomo to RGB
        rgb = np.stack([tomo_n, tomo_n, tomo_n], axis=-1)  # (ny, nx, 3)

        # Overlay mask in colour where mask > 0
        mask_bin = (mask_img > 0).astype(np.float32)
        for c, cv in enumerate(color):
            rgb[:, :, c] = rgb[:, :, c] * (1 - alpha * mask_bin) + cv * alpha * mask_bin

        rgb = np.clip(rgb, 0, 1)

        dpi  = 100
        figw = max(4.0, nx / dpi)
        figh = max(3.0, ny / dpi)
        fig  = Figure(figsize=(figw, figh), dpi=dpi)
        ax   = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb, origin='lower', aspect='equal', interpolation='bilinear')
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Picks overlay
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_COLS = ('rlnLCCmax', 'rlnMaxValueProbDistribution', 'rlnScore', 'score')


def slab_with_picks_b64(
    mrc_path,
    df,
    slab_angst: float = 300.0,
    particle_diameter: float = None,
    pct: tuple = (1, 99),
) -> Optional[dict]:
    """
    Render the central Z slab of a tomogram with particle positions overlaid
    as circles, encoded as a base64 PNG.

    Circles are coloured by score when a score column is present in df
    (rlnLCCmax for pytom-match; rlnMaxValueProbDistribution for gapstop).
    Falls back to flat yellow if no score column is found.

    Parameters
    ----------
    mrc_path          : path-like
    df                : pandas DataFrame — RELION5 (rlnCenteredCoordinate*Angst),
                        RELION3/4 (rlnCoordinateX/Y/Z pixels), or any df with
                        those columns plus an optional score column.
    slab_angst        : total slab thickness in Å (default 300 Å)
    particle_diameter : particle diameter in Å — sets circle radius.
                        If None, draws small cross markers instead.
    pct               : (lo, hi) percentiles for contrast normalisation

    Returns
    -------
    dict(img_b64, n_total, n_shown) or None on failure
    """
    try:
        import numpy as np
        import mrcfile
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle
    except ImportError:
        return None

    # ── Load central slab ────────────────────────────────────────────────────
    try:
        with mrcfile.mmap(str(mrc_path), mode='r', permissive=True) as mrc:
            vox = float(mrc.voxel_size.x) or 1.0
            nz, ny, nx = mrc.data.shape
            hp  = max(1, int(round((slab_angst / 2.0) / vox)))
            zc  = nz // 2
            zs  = max(0,  zc - hp)
            ze  = min(nz, zc + hp)
            slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)
    except Exception:
        return None

    img = slab.mean(axis=0)   # (ny, nx)

    # ── Particle coordinates ─────────────────────────────────────────────────
    # Support both RELION5 (centered Angst) and RELION3/4 (pixel coords)
    r5_cols = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
               'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    r4_cols = {'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'}

    if r5_cols.issubset(df.columns):
        px_size = float(df['rlnTomoTiltSeriesPixelSize'].iloc[0])
        x_px = df['rlnCenteredCoordinateXAngst'] / px_size + nx / 2
        y_px = df['rlnCenteredCoordinateYAngst'] / px_size + ny / 2
        z_px = df['rlnCenteredCoordinateZAngst'] / px_size + nz / 2
    elif r4_cols.issubset(df.columns):
        # Coordinates are already in pixels (0-indexed)
        px_size = vox   # use MRC header pixel size for circle radius
        x_px = df['rlnCoordinateX']
        y_px = df['rlnCoordinateY']
        z_px = df['rlnCoordinateZ']
    else:
        return None

    # Keep only particles within the slab Z range
    in_slab  = (z_px >= zs) & (z_px <= ze)
    x_shown  = x_px[in_slab].values
    y_shown  = y_px[in_slab].values
    n_total  = len(df)
    n_shown  = int(in_slab.sum())

    # Detect score column for colour-mapping
    score_col = next((c for c in _SCORE_COLS if c in df.columns), None)
    scores_shown = df[score_col][in_slab].values if score_col is not None else None

    # ── Plot ─────────────────────────────────────────────────────────────────
    from matplotlib import cm as _cm
    from matplotlib.colors import Normalize as _Normalize
    from matplotlib.colorbar import ColorbarBase as _ColorbarBase

    p_lo, p_hi = np.percentile(img, pct)

    has_scores  = scores_shown is not None and n_shown > 0
    has_circles = particle_diameter is not None

    # Leave right margin for colorbar when scores are present
    cb_frac = 0.07 if has_scores else 0.0
    dpi  = 100
    figw = max(4.0, nx / dpi) + (0.4 if has_scores else 0.0)
    figh = max(3.0, ny / dpi)
    fig  = Figure(figsize=(figw, figh), dpi=dpi)

    img_right = 1.0 - cb_frac - (0.01 if has_scores else 0.0)
    ax = fig.add_axes([0, 0, img_right, 1.0])
    ax.imshow(img, cmap='gray', vmin=p_lo, vmax=p_hi,
              origin='lower', aspect='equal', interpolation='bilinear')

    if n_shown > 0:
        if has_scores:
            s_lo, s_hi = float(scores_shown.min()), float(scores_shown.max())
            if s_lo == s_hi:
                s_hi = s_lo + 1e-6
            cmap_picks = _cm.get_cmap('plasma')
            norm_picks = _Normalize(vmin=s_lo, vmax=s_hi)
            colors = [cmap_picks(norm_picks(s)) for s in scores_shown]
        else:
            colors = ['#ffdd00'] * n_shown

        if has_circles:
            radius_px = (particle_diameter / 2.0) / px_size
            for x, y, c in zip(x_shown, y_shown, colors):
                ax.add_patch(Circle(
                    (x, y), radius=radius_px,
                    fill=False, edgecolor=c,
                    linewidth=0.8, alpha=0.8,
                ))
        else:
            for x, y, c in zip(x_shown, y_shown, colors):
                ax.plot(x, y, '+', color=c, markersize=6,
                        markeredgewidth=0.8, alpha=0.8)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.axis('off')

    # Colorbar
    if has_scores:
        cax = fig.add_axes([img_right + 0.01, 0.1, 0.025, 0.8])
        cb  = _ColorbarBase(cax, cmap=_cm.get_cmap('plasma'),
                            norm=_Normalize(vmin=float(scores_shown.min()),
                                            vmax=float(scores_shown.max())),
                            orientation='vertical')
        cb.set_label(score_col, fontsize=6)
        cax.tick_params(labelsize=5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)

    return {
        'img_b64': base64.b64encode(buf.read()).decode(),
        'n_total': n_total,
        'n_shown': n_shown,
        'slab_a':  (ze - zs) * vox,
        'vox':     vox,
    }


def make_picks_html(
    entries: list,
    out_path,
    title: str,
    command: str,
    slab_angst: float = 300.0,
) -> None:
    """
    Write a standalone HTML report showing per-TS central-slab projections
    with particle picks overlaid.

    Parameters
    ----------
    entries : list of dicts with keys:
        ts_name    — str
        img_b64    — base64 PNG or None
        n_total    — int, total particles extracted
        n_shown    — int, particles visible in the slab
        tomo_path  — str
        metadata   — dict of extra key/value pairs (optional)
    out_path   : path-like
    title      : page title / h1 heading
    command    : command string to display
    slab_angst : slab thickness used
    """
    out_path = Path(out_path)

    def _img_panel(b64, label, title=''):
        if b64:
            return (
                f'<div class="panel">'
                f'<div class="panel-label">{label}</div>'
                f'<img src="data:image/png;base64,{b64}" title="{title}" alt="{label}">'
                f'</div>'
            )
        return (
            f'<div class="panel">'
            f'<div class="panel-label">{label}</div>'
            f'<div class="img-placeholder">unavailable</div>'
            f'</div>'
        )

    cards_html = []
    for e in entries:
        ts      = e['ts_name']
        n_total = e.get('n_total', '?')
        n_shown = e.get('n_shown', '?')
        meta    = e.get('metadata', {})

        tomo_panel  = _img_panel(e.get('img_b64'),   'Tomogram (picks)', e.get('tomo_path', ''))
        score_panel = _img_panel(e.get('score_b64'), 'Score map', '')

        meta_html = ''
        if meta:
            rows = ''.join(
                f'<tr><td class="mk">{k}</td><td class="mv">{v}</td></tr>'
                for k, v in meta.items()
            )
            meta_html = f'<table class="meta">{rows}</table>'

        cards_html.append(f'''
  <div class="card">
    <div class="card-header">
      <span class="card-title">{ts}</span>
      <span class="card-count">{n_shown} / {n_total} particles in slab</span>
    </div>
    {meta_html}
    <div class="pair">
      {tomo_panel}
      {score_panel}
    </div>
  </div>''')

    cards_block = '\n'.join(cards_html)
    n_ok = sum(1 for e in entries if e.get('img_b64'))

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 20px;
  }}
  h1 {{ font-size: 1.4em; color: #a8d8ea; margin-bottom: 6px; }}
  .subtitle {{ color: #888; font-size: 0.85em; margin-bottom: 18px; }}
  .cmd-block {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; font-family: monospace; font-size: 0.82em;
    color: #79c0ff; white-space: pre-wrap; word-break: break-all;
    margin-bottom: 24px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(700px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 14px; overflow: hidden;
  }}
  .card-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 8px;
  }}
  .card-title {{ font-weight: 600; font-size: 0.95em; color: #a8d8ea; }}
  .card-count {{ font-size: 0.78em; color: #f4c542; }}
  .meta {{ border-collapse: collapse; margin-bottom: 10px; font-size: 0.78em; }}
  .meta td {{ padding: 1px 10px 1px 0; }}
  .meta .mk {{ color: #888; }}
  .meta .mv {{ color: #ccc; }}
  .pair {{ display: flex; gap: 8px; }}
  .panel {{ flex: 1; min-width: 0; }}
  .panel-label {{
    font-size: 0.72em; color: #888; text-align: center;
    margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .panel img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  .img-placeholder {{
    width: 100%; aspect-ratio: 4/3; background: #0d1117; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: #555; font-size: 0.8em;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  Central {slab_angst:.0f} Å slab &nbsp;·&nbsp;
  circles = particle positions within slab &nbsp;·&nbsp;
  {n_ok} tomograms &nbsp;·&nbsp; {timestamp}
</div>
<div class="cmd-block">{command}</div>
<div class="grid">
{cards_block}
</div>
</body>
</html>
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f'Picks QC report: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Interactive picks QC (dev)
# ─────────────────────────────────────────────────────────────────────────────

def slab_picks_data(
    mrc_path,
    df,
    slab_angst: float = 300.0,
    particle_diameter: float = None,
    max_px: int = 900,
) -> Optional[dict]:
    """
    Return raw data for the interactive Canvas-based picks overlay.

    The background slab is rendered as a grayscale PNG (capped at max_px).
    Particle coordinates are returned in PNG pixel space (already scaled).

    Returns dict or None on failure.
    """
    try:
        import numpy as np
        import mrcfile
    except ImportError:
        return None

    try:
        with mrcfile.mmap(str(mrc_path), mode='r', permissive=True) as mrc:
            vox = float(mrc.voxel_size.x) or 1.0
            nz, ny, nx = mrc.data.shape
            hp   = max(1, int(round((slab_angst / 2.0) / vox)))
            zc   = nz // 2
            zs   = max(0,  zc - hp)
            ze   = min(nz, zc + hp)
            slab = np.asarray(mrc.data[zs:ze], dtype=np.float32)
    except Exception:
        return None

    img = slab.mean(axis=0)   # (ny, nx)

    # Scale factor so the longest side ≤ max_px
    scale = min(1.0, max_px / max(nx, ny)) if max(nx, ny) > max_px else 1.0
    img_nx = max(1, round(nx * scale))
    img_ny = max(1, round(ny * scale))
    # Scale slab Z bounds to PNG space (Z not rescaled, keep as-is)
    slab_scale_z = scale   # same scale applies

    # Render background as grayscale PNG via PIL at scaled size.
    # PIL fromarray treats row 0 as the top of the image; MRC row 0 is the
    # bottom, so flipud before converting to match MRC/matplotlib convention.
    try:
        from PIL import Image as _PIL
        p1, p99 = float(np.percentile(img, 1)), float(np.percentile(img, 99))
        span = (p99 - p1) or 1.0
        arr = np.clip((img - p1) / span * 255, 0, 255).astype(np.uint8)
        pil = _PIL.fromarray(np.flipud(arr), mode='L').resize((img_nx, img_ny), _PIL.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        buf.seek(0)
        bg_b64 = base64.b64encode(buf.read()).decode()
    except Exception:
        bg_b64 = None

    # Parse coordinates (RELION5 or RELION4)
    r5_cols = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
               'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    r4_cols = {'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'}

    if r5_cols.issubset(df.columns):
        px_size = float(df['rlnTomoTiltSeriesPixelSize'].iloc[0])
        x_mrc = (df['rlnCenteredCoordinateXAngst'] / px_size + nx / 2).values
        y_mrc = (df['rlnCenteredCoordinateYAngst'] / px_size + ny / 2).values
        z_mrc = (df['rlnCenteredCoordinateZAngst'] / px_size + nz / 2).values
    elif r4_cols.issubset(df.columns):
        x_mrc = df['rlnCoordinateX'].values.astype(float)
        y_mrc = df['rlnCoordinateY'].values.astype(float)
        z_mrc = df['rlnCoordinateZ'].values.astype(float)
    else:
        return None

    # Scale coordinates to PNG pixel space.
    # Y is flipped to match the flipud'd PIL background (MRC row 0 = canvas bottom).
    x_png = x_mrc * scale
    y_png = img_ny - y_mrc * scale
    z_png = z_mrc  # Z not rescaled (used only for slab bounds comparison)

    score_col = next((c for c in _SCORE_COLS if c in df.columns), None)
    scores    = df[score_col].values.astype(float) if score_col else np.ones(len(df))

    radius_px = (particle_diameter / 2.0) / (vox / scale) if particle_diameter else None

    import json as _json
    particles_json = _json.dumps([
        {'x': round(float(x), 2), 'y': round(float(y), 2),
         'z': round(float(z), 2), 's': round(float(s), 5)}
        for x, y, z, s in zip(x_png, y_png, z_png, scores)
    ])

    return {
        'img_b64':    bg_b64,
        'particles':  particles_json,
        'slab_zs':    float(zs),
        'slab_ze':    float(ze),
        'radius_px':  float(radius_px) if radius_px else None,
        'img_nx':     img_nx,
        'img_ny':     img_ny,
        'score_col':  score_col or 'score',
        'score_min':  float(scores.min()),
        'score_max':  float(scores.max()),
        'n_total':    len(scores),
        'has_scores': score_col is not None,
    }


_PLASMA_STOPS = [
    (0.00, (13,  8,  135)),
    (0.25, (126, 3,  168)),
    (0.50, (204, 71, 120)),
    (0.75, (248, 148, 65)),
    (1.00, (240, 249, 33)),
]

_PLASMA_JS = (
    'function plasmaColor(t){'
    'var s=[' +
    ','.join(f'[{p},[{r},{g},{b}]]' for p, (r, g, b) in _PLASMA_STOPS) +
    '];'
    'for(var i=0;i<s.length-1;i++){'
    'if(t>=s[i][0]&&t<=s[i+1][0]){'
    'var lt=(t-s[i][0])/(s[i+1][0]-s[i][0]);'
    'var c0=s[i][1],c1=s[i+1][1];'
    'return "rgba("+Math.round(c0[0]+lt*(c1[0]-c0[0]))+","'
    '+Math.round(c0[1]+lt*(c1[1]-c0[1]))+","'
    '+Math.round(c0[2]+lt*(c1[2]-c0[2]))+",0.85)";'
    '}}'
    'return "rgba(240,249,33,0.85)";}'
)


def make_picks_html_dev(
    entries: list,
    out_path,
    title: str,
    command: str,
    slab_angst: float = 300.0,
) -> None:
    """
    Interactive HTML QC report with a per-TS score slider.

    Each entry needs 'picks_data' key (from slab_picks_data) in addition
    to the standard make_picks_html keys.  Entries without picks_data fall
    back to the static img_b64 panel.

    A "Download thresholds CSV" button writes ts_name,threshold for every
    tomogram at its current slider position.
    """
    import json as _json

    out_path = Path(out_path)

    cards = []
    for idx, e in enumerate(entries):
        ts   = e['ts_name']
        tid  = f'ts{idx}'
        pd   = e.get('picks_data')
        meta = e.get('metadata', {})

        meta_html = ''
        if meta:
            rows = ''.join(
                f'<tr><td class="mk">{k}</td><td class="mv">{v}</td></tr>'
                for k, v in meta.items()
            )
            meta_html = f'<table class="meta">{rows}</table>'

        if pd and pd.get('img_b64'):
            s_min   = pd['score_min']
            s_max   = pd['score_max']
            s_init  = s_max
            s_step  = round((s_max - s_min) / 1000, 6) or 0.0001
            img_nx  = pd['img_nx']
            img_ny  = pd['img_ny']
            r_px    = pd['radius_px'] if pd['radius_px'] else 'null'
            zs      = pd['slab_zs']
            ze      = pd['slab_ze']
            sc_col  = pd['score_col']
            n_total = pd['n_total']
            has_sc  = str(pd['has_scores']).lower()
            p_json  = pd['particles']   # already JSON string

            canvas_block = f"""
<div class="canvas-wrap" style="position:relative;display:inline-block;line-height:0;width:100%">
  <img id="bg_{tid}" src="data:image/png;base64,{pd['img_b64']}"
       style="display:block;width:100%;height:auto">
  <canvas id="cv_{tid}" width="{img_nx}" height="{img_ny}"
          style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none">
  </canvas>
</div>
<div class="ctrl">
  <label>{sc_col} threshold:
    <input type="range" class="ts-slider" data-ts="{ts}"
           id="sl_{tid}" min="{s_min:.6f}" max="{s_max:.6f}"
           step="{s_step}" value="{s_init}"
           oninput="onSliderInput('{tid}','{ts}',parseFloat(this.value))">
    <span id="tv_{tid}">{s_init:.4f}</span>
  </label>
  <span id="cnt_{tid}" class="cnt">? / {n_total}</span>
  <button class="fs-btn" onclick="openFullscreen('{tid}','{ts}')">&#x2922; fullscreen</button>
</div>
<script>
(function(){{
  var D={{particles:{p_json},zs:{zs},ze:{ze},r:{r_px},img_nx:{img_nx},img_ny:{img_ny},
          smin:{s_min},smax:{s_max},hasSc:{has_sc}}};
  window._pd=window._pd||{{}};window._pd['{tid}']=D;
  drawPicks('{tid}',{s_init});
}})();
</script>"""
        else:
            # Fallback: static image
            b64 = e.get('img_b64', '')
            canvas_block = (
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'
                if b64 else '<div class="img-placeholder">unavailable</div>'
            )

        score_panel = ''
        if e.get('score_b64'):
            score_panel = (
                f'<div class="panel">'
                f'<div class="panel-label">Score map</div>'
                f'<img src="data:image/png;base64,{e["score_b64"]}" style="max-width:100%">'
                f'</div>'
            )

        cards.append(f"""
<div class="card">
  <div class="card-header">{ts}
    <span class="n-particles">{e.get('n_total','?')} particles</span>
  </div>
  <div class="panels">
    <div class="panel">
      <div class="panel-label">Tomogram — picks (score coloured)</div>
      {canvas_block}
    </div>
    {score_panel}
  </div>
  {meta_html}
</div>""")

    ts_names_js = _json.dumps([e['ts_name'] for e in entries])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>{title} (interactive)</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background:#0a0a1a; color:#e0e0e0; font-family:monospace; font-size:13px; padding:16px; }}
h1 {{ color:#7eb8f7; margin-bottom:4px; }}
.cmd {{ background:#111; color:#aaa; padding:8px; border-radius:4px;
        font-size:11px; margin-bottom:12px; word-break:break-all; }}
.top-bar {{ margin-bottom:14px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.btn {{ background:#1e3a5f; color:#7eb8f7; border:1px solid #2a5080; padding:6px 14px;
        border-radius:4px; cursor:pointer; font-size:12px; font-family:monospace; }}
.btn:hover {{ background:#2a5080; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(700px,1fr)); gap:16px; }}
.card {{ background:#16213e; border:1px solid #2a2a4a; border-radius:8px; padding:14px; }}
.card-header {{ font-size:14px; font-weight:bold; color:#7eb8f7; margin-bottom:10px; }}
.n-particles {{ color:#aaa; font-size:11px; margin-left:10px; }}
.panels {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-start; }}
.panel {{ flex:1 1 0; min-width:280px; }}
.panel-label {{ font-size:11px; color:#888; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center; }}
.panel img {{ width:100%; height:auto; display:block; }}
.fs-btn {{ background:none; border:1px solid #3a3a6a; color:#7eb8f7; padding:1px 7px;
           border-radius:3px; cursor:pointer; font-size:11px; font-family:monospace; }}
.fs-btn:hover {{ background:#1e3a5f; }}
.img-placeholder {{ background:#111; height:200px; display:flex;
                    align-items:center; justify-content:center; color:#444; }}
.canvas-wrap {{ width:100%; }}
.ctrl {{ margin-top:8px; display:flex; flex-wrap:wrap; align-items:center; gap:12px; }}
.ctrl label {{ display:flex; align-items:center; gap:8px; flex:1; }}
.ctrl input[type=range] {{ flex:1; accent-color:#f89441; }}
.cnt {{ color:#f89441; font-size:11px; white-space:nowrap; }}
table.meta {{ margin-top:8px; border-collapse:collapse; width:100%; }}
table.meta td {{ padding:2px 6px; font-size:11px; }}
td.mk {{ color:#888; }}
td.mv {{ color:#ccc; }}
</style>
</head>
<body>
<h1>{title} <small style="font-size:12px;color:#666">(interactive dev)</small></h1>
<div class="cmd">{command}</div>
<div class="top-bar">
  <span style="color:#888;font-size:11px">Slab: {slab_angst:.0f} Å</span>
  <button class="btn" onclick="downloadCSV()">Save thresholds CSV</button>
  <button class="btn" onclick="resetAll()">Reset all thresholds</button>
</div>
<div class="grid">
{''.join(cards)}
</div>
<script>
{_PLASMA_JS}

function drawPicks(tid, thresh) {{
  var D = window._pd && window._pd[tid];
  if (!D) return;
  var cv = document.getElementById('cv_'+tid);
  if (!cv) return;
  var ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  var cnt=0, ps=D.particles;
  var smin=D.smin, smax=D.smax, rng=smax-smin||1e-9;
  for (var i=0;i<ps.length;i++) {{
    var p=ps[i];
    if (p.z<D.zs || p.z>D.ze) continue;
    if (p.s<thresh) continue;
    var col = D.hasSc ? plasmaColor((p.s-smin)/rng) : 'rgba(255,221,0,0.8)';
    ctx.beginPath();
    if (D.r) {{
      ctx.arc(p.x,p.y,D.r,0,2*Math.PI);
      ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.globalAlpha=0.9;
      ctx.stroke();
    }} else {{
      ctx.moveTo(p.x-4,p.y); ctx.lineTo(p.x+4,p.y);
      ctx.moveTo(p.x,p.y-4); ctx.lineTo(p.x,p.y+4);
      ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.globalAlpha=0.9;
      ctx.stroke();
    }}
    cnt++;
  }}
  var tv=document.getElementById('tv_'+tid);
  if(tv) tv.textContent=thresh.toFixed(5);
  var ce=document.getElementById('cnt_'+tid);
  if(ce) ce.textContent=cnt+' / '+ps.length;
}}

function downloadCSV() {{
  var names = {ts_names_js};
  var lines = ['ts_name,threshold'];
  document.querySelectorAll('.ts-slider').forEach(function(sl) {{
    lines.push(sl.dataset.ts + ',' + parseFloat(sl.value).toFixed(5));
  }});
  var blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'thresholds.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}}

function resetAll() {{
  document.querySelectorAll('.ts-slider').forEach(function(sl) {{
    var min=parseFloat(sl.min), max=parseFloat(sl.max);
    sl.value = max;
    var tid = sl.id.replace('sl_','');
    onSliderInput(tid, sl.dataset.ts, parseFloat(sl.value));
  }});
}}

function onSliderInput(tid, ts, val) {{
  drawPicks(tid, val);
  try {{ localStorage.setItem('gs_thr_'+ts, String(val)); }} catch(e) {{}}
}}

function syncThreshold(tid, ts, val) {{
  var sl = document.getElementById('sl_'+tid);
  if (!sl) return;
  sl.value = val;
  drawPicks(tid, val);
}}

window.addEventListener('storage', function(e) {{
  if (!e.key || e.key.indexOf('gs_thr_') !== 0) return;
  var ts = e.key.slice(7);
  document.querySelectorAll('.ts-slider').forEach(function(sl) {{
    if (sl.dataset.ts !== ts) return;
    var tid = sl.id.replace('sl_','');
    sl.value = e.newValue;
    drawPicks(tid, parseFloat(e.newValue));
  }});
}});

window._plasmaColor = plasmaColor;

function openFullscreen(tid, ts) {{
  var D = window._pd && window._pd[tid];
  if (!D) return;
  var sl = document.getElementById('sl_'+tid);
  var curVal = sl ? parseFloat(sl.value) : D.smax;
  var bgEl = document.getElementById('bg_'+tid);
  var imgSrc = bgEl ? bgEl.src : '';
  var smin = D.smin, smax = D.smax;
  var step = Math.round((smax-smin)/1000*1e6)/1e6 || 0.0001;
  var storeKey = 'gs_thr_'+ts;
  var w = window.open('','_blank');
  w.document.write(
    '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>'+ts+' \u2014 picks<\/title><style>' +
    '*{{box-sizing:border-box;margin:0;padding:0}}' +
    'body{{background:#000;display:flex;flex-direction:column;height:100vh;overflow:hidden}}' +
    '.wrap{{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden}}' +
    '.inner{{position:relative;display:inline-block;line-height:0}}' +
    '.inner img{{max-width:100vw;max-height:calc(100vh - 52px);display:block}}' +
    '.inner canvas{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}}' +
    '.ctrl{{height:52px;padding:0 16px;background:#111;border-top:1px solid #222;' +
           'display:flex;gap:12px;align-items:center}}' +
    '.lbl{{color:#7eb8f7;font-size:13px;font-family:monospace}}' +
    'input[type=range]{{flex:1;accent-color:#f89441}}' +
    '.tv{{color:#ccc;font-size:12px;font-family:monospace;min-width:70px}}' +
    '.cnt{{color:#f89441;font-size:12px;font-family:monospace;white-space:nowrap}}' +
    '<\/style><\/head><body>' +
    '<div class="wrap"><div class="inner">' +
    '<img id="bg" src="'+imgSrc+'">' +
    '<canvas id="cv" width="'+D.img_nx+'" height="'+D.img_ny+'"><\/canvas>' +
    '<\/div><\/div>' +
    '<div class="ctrl">' +
    '<span class="lbl">'+ts+'<\/span>' +
    '<input type="range" id="sl" min="'+smin+'" max="'+smax+'" step="'+step+'" value="'+curVal+'" oninput="onSlide(parseFloat(this.value))">' +
    '<span class="tv" id="tv">'+curVal.toFixed(5)+'<\/span>' +
    '<span class="cnt" id="cnt">...<\/span>' +
    '<\/div>' +
    '<script>' +
    'function plasmaColor(t){{return window.opener&&window.opener._plasmaColor?window.opener._plasmaColor(t):"rgba(255,221,0,0.8)";}}' +
    'var D=window.opener._pd["'+tid+'"];' +
    'var SKEY="'+storeKey+'";' +
    'function drawPicks(thresh){{' +
    '  var cv=document.getElementById("cv");if(!cv||!D)return;' +
    '  var ctx=cv.getContext("2d");ctx.clearRect(0,0,cv.width,cv.height);' +
    '  var cnt=0,ps=D.particles,smin=D.smin,smax=D.smax,rng=smax-smin||1e-9;' +
    '  for(var i=0;i<ps.length;i++){{' +
    '    var p=ps[i];if(p.z<D.zs||p.z>D.ze||p.s<thresh)continue;' +
    '    var col=D.hasSc?plasmaColor((p.s-smin)/rng):"rgba(255,221,0,0.8)";' +
    '    ctx.beginPath();' +
    '    if(D.r){{ctx.arc(p.x,p.y,D.r,0,2*Math.PI);ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.globalAlpha=0.9;ctx.stroke();}}' +
    '    else{{ctx.moveTo(p.x-4,p.y);ctx.lineTo(p.x+4,p.y);ctx.moveTo(p.x,p.y-4);ctx.lineTo(p.x,p.y+4);ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.globalAlpha=0.9;ctx.stroke();}}' +
    '    cnt++;' +
    '  }}' +
    '  var tv=document.getElementById("tv");if(tv)tv.textContent=thresh.toFixed(5);' +
    '  var ce=document.getElementById("cnt");if(ce)ce.textContent=cnt+" / "+ps.length;' +
    '}}' +
    'function onSlide(val){{' +
    '  drawPicks(val);' +
    '  try{{localStorage.setItem(SKEY,String(val));}}catch(e){{}}' +
    '  if(window.opener&&!window.opener.closed)window.opener.syncThreshold("'+tid+'","'+ts+'",val);' +
    '}}' +
    'window.addEventListener("storage",function(e){{' +
    '  if(e.key===SKEY){{var v=parseFloat(e.newValue);document.getElementById("sl").value=v;drawPicks(v);}}' +
    '}});' +
    'drawPicks('+curVal+');' +
    '<\/script><\/body><\/html>'
  );
  w.document.close();
}}
</script>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f'Picks QC report (interactive): {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Orthoslice panels with mask overlay
# ─────────────────────────────────────────────────────────────────────────────

def orthoslices_with_mask_b64(
    tomo_path,
    mask_path,
    color: tuple = (0.4, 0.6, 1.0),
    alpha: float = 0.45,
    pct: tuple = (1, 99),
    gap_px: int = 8,
) -> Optional[str]:
    """
    Render three orthogonal central slices (XY, XZ, YZ) of a tomogram with an
    optional mask overlaid as a semi-transparent coloured layer.

    Layout (all at the same pixel scale):
      XZ panel  — top-left    (x horizontal, z vertical)
      XY panel  — bottom-left (x horizontal, y vertical)
      YZ panel  — right       (y horizontal, z vertical)
      (top-right corner is blank)

    Parameters
    ----------
    tomo_path : path-like — tomogram MRC
    mask_path : path-like or None — binary mask MRC (or None for no overlay)
    color     : RGB tuple 0–1 for overlay colour (default light blue)
    alpha     : overlay opacity 0–1 (default 0.45)
    pct       : (lo, hi) percentiles for tomogram contrast
    gap_px    : blank gap in data-pixel units between panels (default 8)

    Returns
    -------
    base64-encoded PNG str, or None on failure
    """
    try:
        import numpy as np
        import mrcfile
        from matplotlib.figure import Figure
    except ImportError:
        return None

    def _resize_slice(sl, target_shape):
        """Resize a 2-D array to target_shape using nearest-neighbour."""
        if sl.shape == target_shape:
            return sl
        from PIL import Image as _PIL
        # PIL resize: (width, height) = (cols, rows)
        return np.array(
            _PIL.fromarray(sl).resize(
                (target_shape[1], target_shape[0]), _PIL.NEAREST
            )
        )

    def _overlay(gray_n, mask_sl, target_shape):
        """Return RGB array with mask overlaid on normalised grayscale slice."""
        if mask_sl is not None:
            m = _resize_slice(mask_sl, target_shape).astype(np.float32)
            m = (m > 0).astype(np.float32)
        rgb = np.stack([gray_n, gray_n, gray_n], axis=-1)
        if mask_sl is not None:
            for c, cv in enumerate(color):
                rgb[:, :, c] = rgb[:, :, c] * (1 - alpha * m) + cv * alpha * m
        return np.clip(rgb, 0, 1)

    try:
        with mrcfile.mmap(str(tomo_path), mode='r', permissive=True) as mrc:
            vox = float(mrc.voxel_size.x) or 1.0
            nz, ny, nx = mrc.data.shape
            zc, yc, xc = nz // 2, ny // 2, nx // 2
            sl_xy = np.asarray(mrc.data[zc, :, :],  dtype=np.float32)  # (ny, nx)
            sl_xz = np.asarray(mrc.data[:, yc, :],  dtype=np.float32)  # (nz, nx)
            sl_yz = np.asarray(mrc.data[:, :, xc],  dtype=np.float32)  # (nz, ny)

        # Shared contrast from XY slice
        p_lo, p_hi = np.percentile(sl_xy, pct)
        span = max(float(p_hi - p_lo), 1e-6)

        def _norm(sl):
            return np.clip((sl.astype(np.float32) - p_lo) / span, 0, 1)

        # Load mask slices (optional)
        m_xy = m_xz = m_yz = None
        if mask_path is not None:
            try:
                with mrcfile.mmap(str(mask_path), mode='r', permissive=True) as mrc:
                    mnz, mny, mnx = mrc.data.shape
                    mzc = int(round(zc * mnz / nz))
                    myc = int(round(yc * mny / ny))
                    mxc = int(round(xc * mnx / nx))
                    mzc = min(mzc, mnz - 1)
                    myc = min(myc, mny - 1)
                    mxc = min(mxc, mnx - 1)
                    m_xy = np.asarray(mrc.data[mzc, :, :], dtype=np.float32)
                    m_xz = np.asarray(mrc.data[:, myc, :], dtype=np.float32)
                    m_yz = np.asarray(mrc.data[:, :, mxc], dtype=np.float32)
            except Exception:
                pass

        rgb_xy = _overlay(_norm(sl_xy), m_xy, (ny, nx))
        rgb_xz = _overlay(_norm(sl_xz), m_xz, (nz, nx))
        rgb_yz = _overlay(_norm(sl_yz), m_yz, (nz, ny))

        # Figure layout: all panels at the same pixel scale
        # W = nx + gap + ny  (data pixels)
        # H = nz + gap + ny
        gap = gap_px
        W = nx + gap + ny
        H = nz + gap + ny

        dpi  = 100
        figw = max(4.0, W / dpi)
        figh = max(3.0, H / dpi)
        fig  = Figure(figsize=(figw, figh), dpi=dpi)
        fig.patch.set_facecolor('#111111')

        # Axes positions in normalised figure coordinates [left, bottom, width, height]
        ax_xy = fig.add_axes([0,               0,            nx / W, ny / H])
        ax_xz = fig.add_axes([0,               (ny + gap) / H, nx / W, nz / H])
        ax_yz = fig.add_axes([(nx + gap) / W,  0,            ny / W, nz / H])

        for ax, rgb, origin in [
            (ax_xy, rgb_xy, 'lower'),
            (ax_xz, rgb_xz, 'lower'),
            (ax_yz, rgb_yz, 'lower'),
        ]:
            ax.imshow(rgb, origin=origin, aspect='auto', interpolation='nearest')
            ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches=None, pad_inches=0,
                    dpi=dpi, facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception:
        return None


def make_ortho_html(
    entries: list,
    out_path,
    title: str,
    command: str,
) -> None:
    """
    Write a standalone HTML report showing per-TS orthogonal section panels
    (XY, XZ, YZ) with mask overlay.

    Parameters
    ----------
    entries : list of dicts with keys:
        ts_name    — str
        img_b64    — base64 PNG from orthoslices_with_mask_b64 (or None)
        tomo_path  — str
        mask_path  — str
        metadata   — dict of extra key/value pairs (optional)
    out_path   : path-like
    title      : page title / h1 heading
    command    : command string to display
    """
    out_path = Path(out_path)

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    n_ok = sum(1 for e in entries if e.get('img_b64'))

    cards_html = []
    for e in entries:
        ts   = e['ts_name']
        b64  = e.get('img_b64')
        meta = e.get('metadata', {})

        if b64:
            img_html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{ts}" style="width:100%;height:auto;display:block;border-radius:4px;">'
            )
        else:
            img_html = (
                '<div class="img-placeholder">unavailable</div>'
            )

        meta_html = ''
        if meta:
            rows = ''.join(
                f'<tr><td class="mk">{k}</td><td class="mv">{v}</td></tr>'
                for k, v in meta.items()
            )
            meta_html = f'<table class="meta">{rows}</table>'

        cards_html.append(f'''
  <div class="card">
    <div class="card-title">{ts}</div>
    {meta_html}
    <div class="ortho-label">
      XZ (top) · XY (bottom-left) · YZ (bottom-right)
    </div>
    {img_html}
  </div>''')

    cards_block = '\n'.join(cards_html)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 20px;
  }}
  h1 {{ font-size: 1.4em; color: #a8d8ea; margin-bottom: 6px; }}
  .subtitle {{ color: #888; font-size: 0.85em; margin-bottom: 18px; }}
  .cmd-block {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; font-family: monospace; font-size: 0.82em;
    color: #79c0ff; white-space: pre-wrap; word-break: break-all;
    margin-bottom: 24px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 14px; overflow: hidden;
  }}
  .card-title {{
    font-weight: 600; font-size: 0.95em; color: #a8d8ea; margin-bottom: 8px;
  }}
  .ortho-label {{
    font-size: 0.72em; color: #888; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .meta {{ border-collapse: collapse; margin-bottom: 10px; font-size: 0.78em; }}
  .meta td {{ padding: 1px 10px 1px 0; }}
  .meta .mk {{ color: #888; }}
  .meta .mv {{ color: #ccc; }}
  .img-placeholder {{
    width: 100%; aspect-ratio: 4/3; background: #0d1117; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: #555; font-size: 0.8em;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  Orthogonal central sections with mask overlay (light blue) &nbsp;·&nbsp;
  {n_ok} tomograms &nbsp;·&nbsp; {timestamp}
</div>
<div class="cmd-block">{command}</div>
<div class="grid">
{cards_block}
</div>
</body>
</html>
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f'Ortho QC report: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_html(
    entries: list,
    out_path,
    title: str,
    command: str,
    before_label: str = 'Before',
    after_label: str  = 'After',
    slab_angst: float = 300.0,
) -> None:
    """
    Write a standalone HTML report with side-by-side before/after projections.

    Parameters
    ----------
    entries : list of dicts, each with keys:
        ts_name        — str
        before_b64     — base64 PNG str (or None if unavailable)
        after_b64      — base64 PNG str (or None if unavailable)
        before_path    — str, original volume path
        after_path     — str, output volume path
        metadata       — dict of extra key/value pairs to display (optional)
    out_path     : path-like — output HTML file
    title        : page title / h1 heading
    command      : the command string to display
    before_label : label for the left image column
    after_label  : label for the right image column
    slab_angst   : slab thickness used (for caption)
    """
    out_path = Path(out_path)

    # ── Build card HTML for each TS ──────────────────────────────────────────
    cards_html = []
    for e in entries:
        ts   = e['ts_name']
        meta = e.get('metadata', {})

        def _img(b64, label, path):
            if b64:
                return (
                    f'<div class="img-wrap">'
                    f'<div class="img-label">{label}</div>'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'     title="{path}" alt="{label}">'
                    f'</div>'
                )
            return (
                f'<div class="img-wrap missing">'
                f'<div class="img-label">{label}</div>'
                f'<div class="img-placeholder">unavailable</div>'
                f'</div>'
            )

        meta_html = ''
        if meta:
            rows = ''.join(
                f'<tr><td class="mk">{k}</td><td class="mv">{v}</td></tr>'
                for k, v in meta.items()
            )
            meta_html = f'<table class="meta">{rows}</table>'

        cards_html.append(f'''
  <div class="card">
    <div class="card-title">{ts}</div>
    {meta_html}
    <div class="pair">
      {_img(e.get("before_b64"), before_label, e.get("before_path",""))}
      {_img(e.get("after_b64"),  after_label,  e.get("after_path",""))}
    </div>
  </div>''')

    cards_block = '\n'.join(cards_html)
    n_ok     = sum(1 for e in entries if e.get('after_b64'))
    n_fail   = len(entries) - n_ok

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 20px;
  }}
  h1 {{ font-size: 1.4em; color: #a8d8ea; margin-bottom: 6px; }}
  .subtitle {{ color: #888; font-size: 0.85em; margin-bottom: 18px; }}
  .cmd-block {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 12px 16px; font-family: monospace; font-size: 0.82em;
    color: #79c0ff; white-space: pre-wrap; word-break: break-all;
    margin-bottom: 24px;
  }}
  .stats {{
    margin-bottom: 20px; font-size: 0.88em; color: #aaa;
  }}
  .stats span {{ color: #e0e0e0; font-weight: 600; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px;
    padding: 14px; overflow: hidden;
  }}
  .card-title {{
    font-weight: 600; font-size: 0.95em; color: #a8d8ea;
    margin-bottom: 8px;
  }}
  .meta {{ border-collapse: collapse; margin-bottom: 10px; font-size: 0.78em; }}
  .meta td {{ padding: 1px 10px 1px 0; color: #aaa; }}
  .meta .mk {{ color: #888; }}
  .meta .mv {{ color: #ccc; }}
  .pair {{
    display: flex; gap: 8px;
  }}
  .img-wrap {{
    flex: 1; min-width: 0;
  }}
  .img-label {{
    font-size: 0.75em; color: #888; text-align: center;
    margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .img-wrap img {{
    width: 100%; height: auto; display: block;
    border-radius: 4px;
  }}
  .img-placeholder {{
    width: 100%; aspect-ratio: 4/3;
    background: #0d1117; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    color: #555; font-size: 0.8em;
  }}
  .missing .img-label {{ color: #555; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  Central {slab_angst:.0f} Å slab projection &nbsp;·&nbsp;
  {n_ok} processed, {n_fail} failed &nbsp;·&nbsp; {timestamp}
</div>
<div class="cmd-block">{command}</div>
<div class="stats">
  Showing <span>{len(entries)}</span> tomogram(s)
</div>
<div class="grid">
{cards_block}
</div>
</body>
</html>
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f'QC report: {out_path}')
