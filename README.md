# aretomo3-editor

Quality-control diagnostics and editing tools for [AreTomo3](https://github.com/czimaginginstitute/AreTomo3) cryo-ET tilt series.

> **Early development** — written with Claude Code, not yet production-ready. Review outputs before relying on them.

---

## Overview

A command-line toolkit with two main workflows:

1. **`check-gain-transform`** — determine the correct AreTomo3 gain correction parameters (`-RotGain` / `-FlipGain`) by testing candidate transformations against your TIFF movies.  Run this **first**, before processing.

2. **`analyse`** — parse AreTomo3 `.aln` output files, compute per-frame overlap and CTF metrics, flag problematic frames, produce diagnostic plots, and generate an interactive HTML report.

Both commands write results to a shared **`aretomo3_project.json`** in the working directory, recording what was run, with what arguments, and key outputs — a lightweight lab notebook for the dataset.

---

## Installation

```bash
git clone https://github.com/frozenfas/aretomo-tilt-series-qc.git
cd aretomo-tilt-series-qc
conda env create -f environment.yml
conda activate aretomo-parse
pip install -e .
```

---

## Typical pipeline

```bash
# 1. Determine correct gain correction parameters
aretomo3-editor check-gain-transform \
    --gain   gain_20260213T101027.mrc \
    --frames frames/ \
    --output gain_check/

# → open gain_check/report.html to review results
# → note the recommended -RotGain / -FlipGain flags

# 2. Run AreTomo3 with the correct gain flags (external step)
#    AreTomo3 ... -RotGain 0 -FlipGain 1 ...

# 3. Analyse alignment results
aretomo3-editor analyse \
    --input          run001/ \
    --output         run001_analysis/ \
    --gain-check-dir gain_check/

# → open run001_analysis/index.html
```

---

## Commands

### `check-gain-transform`

Tests 4 dimension-preserving gain transforms for **K3 rectangular detectors** (TIFF movies + MRC gain).  Scores each transform by the flatness (coefficient of variation) of the gain-corrected average — the correct transform produces the most uniform image.

| Argument | Default | Description |
|---|---|---|
| `--gain` / `-g` | required | Gain MRC file (must be float32, mode 2) |
| `--frames` / `-f` | required | Directory containing `*_fractions.tiff` movies |
| `--output` / `-o` | `gain_check/` | Output directory |
| `--n-acquisitions` / `-n` | `12` | Max acquisition order to include (filter threshold) |
| `--n-movies` / `-N` | `150` | Movies to randomly sample from the filtered set |

Movies are selected by **acquisition order** parsed from the filename (e.g. `Position_1_2_001_14.00_20260213_171849_fractions.tiff`), not by tilt angle.  Early acquisitions have more uniform illumination and are best for evaluating gain correction.

**Validations at startup:**
- Gain MRC must be float32 (mode 2) — aborts if not
- Gain must be rectangular (K3) — aborts with a clear message for square sensors (Falcon support planned)

**Outputs in `--output`:**

| File | Description |
|---|---|
| `results.json` | Best transform, AreTomo3 flags, all CV/SSIM scores |
| `corrected_averages.png` | 2×2 grid of normalised corrected images per transform |
| `cv_vs_nmovies.png` | CV convergence vs movies accumulated per transform |
| `report.html` | Standalone HTML viewer |
| `aretomo3_project.json` | Backup of project state after this run |

See [docs/check-gain-transform.md](docs/check-gain-transform.md) for full details.

---

### `analyse`

Parses all AreTomo3 `.aln` files in a directory, attaches CTF (`_CTF.txt`), dose/tilt (`_TLT.txt`), and metadata (`.mdoc`) data, computes per-frame overlap with the reference frame, flags frames below a threshold, and produces diagnostic plots and an HTML report.

| Argument | Default | Description |
|---|---|---|
| `--input` / `-i` | `run001` | Directory containing `.aln` files |
| `--output` / `-o` | `run001_analysis` | Output directory |
| `--threshold` / `-t` | `80.0` | % overlap below which a frame is flagged |
| `--mdocdir` / `-m` | `frames` | Directory containing `.mdoc` files |
| `--angpix` / `-a` | from mdoc | Pixel size Å/px (for thickness in nm) |
| `--mrcdir` / `-r` | same as `--input` | Directory with raw `.mrc` stacks for header checks |
| `--gain-check-dir` / `-g` | — | Output dir from `check-gain-transform`; adds a Gain Check tab to the HTML report |

**Per-tilt-series diagnostic plot (4 panels):**
- Panel 1 — Overlap % vs corrected tilt angle
- Panel 2 — Spatial frame positions (rectangle overlay)
- Panel 3 — Tilt coverage (symmetric lines through origin, 0° = horizontal)
- Panel 4 — Defocus vs tilt angle with astigmatism error bars and resolution colour coding

**Global summary plot** — 3×3 histogram grid across all tilt series: frame counts, in-plane rotation, alpha offset, defocus, resolution, overlap, astigmatism, CTF CC, flagged frames per TS.

**Sanity checks per tilt series:**
- MRC header nx/ny/nz vs `.aln` width/height/total_frames
- Corrected tilt consistency: `_TLT.txt` nominal + alpha_offset ≈ `.aln` tilt
- TLT z-values link to mdoc ZValues
- All TLT rows accounted for by aligned SECs + dark frame_bs
- mdoc TiltAngle ≈ TLT nominal_tilt

**Outputs in `--output`:**

| File | Description |
|---|---|
| `<ts-name>.png` | 4-panel diagnostic plot per tilt series |
| `global_summary.png` | 3×3 histogram grid for the full dataset |
| `index.html` | Interactive HTML viewer (keyboard navigation, per-TS dropdown; optional Gain Check tab) |
| `alignment_data.json` | All parsed alignment, CTF, dose, and mdoc data |
| `flagged_frames.tsv` | Tab-separated list of frames below the overlap threshold |
| `aretomo3_project.json` | Backup of project state after this run |

---

## Project state file

Both commands maintain `aretomo3_project.json` in the **working directory** — a cumulative record of every run:

```json
{
  "project": { "working_dir": "...", "created": "...", "last_updated": "..." },
  "gain_check": {
    "command":   "aretomo3-editor check-gain-transform --gain ...",
    "args":      { "gain": "...", "n_acquisitions": 12, ... },
    "best_transform": "flipud",
    "aretomo3_rot_gain": 0,
    "aretomo3_flip_gain": 1,
    ...
  },
  "analyse": {
    "command":        "aretomo3-editor analyse --input run001 ...",
    "n_tilt_series":  146,
    "n_flagged_frames": 23,
    ...
  }
}
```

Each command also writes a backup copy to its own output directory.  To revert:

```bash
cp gain_check/aretomo3_project.json .
```

Running from the wrong directory is detected automatically — the command aborts with a clear message showing the expected and current paths.

---

## Package structure

```
aretomo3_editor/
  cli.py                        entry point (aretomo3-editor)
  commands/
    check_gain_transform.py     check-gain-transform subcommand
    analyse.py                  analyse subcommand
    trim_ts.py                  trim-ts subcommand
  shared/
    parsers.py                  .aln / _CTF.txt / _TLT.txt / .mdoc parsers
    geometry.py                 overlap and rotation geometry
    colours.py                  shared colourmap definitions
    project_json.py             shared project state file utilities
docs/
  check-gain-transform.md       full documentation for check-gain-transform
tests/
  test_parsing.py               29+ tests for all parsers
```

---

## Tests

```bash
pytest tests/ -v
```

Tests require the data directory to be mounted at `/mnt/McQueen-002/sconnell/TEST-ARETOMO-PARSE/relion` and are automatically skipped if absent.

---

## Planned features

- EER movie support for `check-gain-transform`
- Square-sensor (Falcon) gain transform search
- RELION / cryoSPARC gain convention variants

---

## License

MIT — see [LICENSE](LICENSE).
