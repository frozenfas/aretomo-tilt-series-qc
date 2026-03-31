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
                        vmin=0, vmax=1)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels([f'{p_lo:.3g}', f'{(p_lo+p_hi)/2:.3g}', f'{p_hi:.3g}'])
        cb.ax.tick_params(labelsize=7, colors='white')
        cb.outline.set_edgecolor('white')
        fig.patch.set_facecolor('#111111')
    else:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_n, cmap=cmap, aspect='equal', interpolation='bilinear',
                  vmin=0, vmax=1)

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
        ax.imshow(rgb, origin='upper', aspect='equal', interpolation='bilinear')
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

    Parameters
    ----------
    mrc_path          : path-like
    df                : pandas DataFrame with RELION5 STAR columns
                        (rlnCenteredCoordinate{X,Y,Z}Angst, rlnTomoTiltSeriesPixelSize)
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
    required = {'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst',
                'rlnCenteredCoordinateZAngst', 'rlnTomoTiltSeriesPixelSize'}
    if not required.issubset(df.columns):
        return None

    px_size  = float(df['rlnTomoTiltSeriesPixelSize'].iloc[0])
    x_px = df['rlnCenteredCoordinateXAngst'] / px_size + nx / 2
    y_px = df['rlnCenteredCoordinateYAngst'] / px_size + ny / 2
    z_px = df['rlnCenteredCoordinateZAngst'] / px_size + nz / 2

    # Keep only particles within the slab Z range
    in_slab  = (z_px >= zs) & (z_px <= ze)
    x_shown  = x_px[in_slab].values
    y_shown  = y_px[in_slab].values
    n_total  = len(df)
    n_shown  = int(in_slab.sum())

    # ── Plot ─────────────────────────────────────────────────────────────────
    p_lo, p_hi = np.percentile(img, pct)
    span = float(p_hi - p_lo) or 1.0

    dpi  = 100
    figw = max(4.0, nx / dpi)
    figh = max(3.0, ny / dpi)
    fig  = Figure(figsize=(figw, figh), dpi=dpi)
    ax   = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap='gray', vmin=p_lo, vmax=p_hi,
              origin='upper', aspect='equal', interpolation='bilinear')

    if n_shown > 0:
        if particle_diameter is not None:
            radius_px = (particle_diameter / 2.0) / px_size
            for x, y in zip(x_shown, y_shown):
                ax.add_patch(Circle(
                    (x, y), radius=radius_px,
                    fill=False, edgecolor='#ffdd00',
                    linewidth=0.8, alpha=0.75,
                ))
        else:
            ax.plot(x_shown, y_shown, '+',
                    color='#ffdd00', markersize=6, markeredgewidth=0.8, alpha=0.75)

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.axis('off')

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
            (ax_xy, rgb_xy, 'upper'),
            (ax_xz, rgb_xz, 'upper'),
            (ax_yz, rgb_yz, 'upper'),
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
